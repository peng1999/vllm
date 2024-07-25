#include "pybind11/detail/common.h"
#define PYBIND11_DETAILED_ERROR_MESSAGES

#include <cstdint>
#include <optional>
#include <random>
#include <chrono>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

namespace py = pybind11;

constexpr float _SAMPLING_EPS = 1e-5;
constexpr int64_t _SEED_0_REPLACEMENT = 3403598558;

int64_t get_num_triton_sampler_splits(int64_t vocab_size) {
  constexpr int64_t MAX_TRITON_N_COLS = 131072;
  return vocab_size / MAX_TRITON_N_COLS;
}

py::array_t<int64_t> pad_tokens(std::vector<py::buffer> tokens, int64_t fill) {
  size_t max_len = 0;
  for (const auto& tokens : tokens) {
    max_len = std::max(max_len, static_cast<size_t>(tokens.request().size));
  }

  // Create and fill prompt_padded_tokens
  py::array_t<int64_t> padded_tokens({tokens.size(), max_len});
  auto buf = padded_tokens.mutable_unchecked<2>();

  for (size_t i = 0; i < tokens.size(); ++i) {
    const auto& subtokens = tokens[i].request();
    assert(subtokens.format == py::format_descriptor<int64_t>::format());
    assert(subtokens.ndim == 1);
    for (size_t j = 0; j < max_len; ++j) {
      if (j < tokens.size()) {
        buf(i, j) = static_cast<int64_t*>(subtokens.ptr)[j];
      } else {
        buf(i, j) = fill;
      }
    }
  }

  return padded_tokens;
}

bool pin_memory = false;

class SamplingTensors {
 public:
  torch::Tensor temperatures;
  torch::Tensor top_ps;
  torch::Tensor top_ks;
  torch::Tensor min_ps;
  torch::Tensor presence_penalties;
  torch::Tensor frequency_penalties;
  torch::Tensor repetition_penalties;
  torch::Tensor sampling_seeds;
  torch::Tensor sample_indices;
  std::optional<torch::Tensor> extra_seeds;
  torch::Tensor prompt_tokens;
  torch::Tensor output_tokens;

  static std::tuple<SamplingTensors, bool, bool, bool> from_sampling_metadata(
      py::object sampling_metadata, int64_t vocab_size,
      const torch::Device& device, py::object dtype_py,
      int64_t extra_seeds_to_generate = 0,
      std::optional<std::vector<py::object>> extra_entropy = std::nullopt) {
    std::vector<py::buffer> prompt_tokens;
    std::vector<py::buffer> output_tokens;
    std::vector<int> top_ks;
    std::vector<float> temperatures;
    std::vector<float> top_ps;
    std::vector<float> min_ps;
    std::vector<float> presence_penalties;
    std::vector<float> frequency_penalties;
    std::vector<float> repetition_penalties;
    std::vector<int64_t> sampling_seeds;
    std::vector<int64_t> sample_indices;
    std::vector<int> prompt_best_of;

    bool do_penalties = false;
    bool do_top_p_top_k = false;
    bool do_min_p = false;

    torch::Dtype dtype = torch::python::detail::py_object_to_dtype(dtype_py);

    int64_t seeds_to_generate =
        extra_seeds_to_generate + get_num_triton_sampler_splits(vocab_size);

    assert(!sampling_metadata.attr("seq_groups").is_none());

    auto seq_groups = sampling_metadata.attr("seq_groups").cast<py::list>();

    for (const auto& seq_group : seq_groups) {
      auto seq_ids = seq_group.attr("seq_ids").cast<std::vector<int64_t>>();
      auto sampling_params = seq_group.attr("sampling_params");
      float temperature = sampling_params.attr("temperature").cast<float>();
      float p = sampling_params.attr("presence_penalty").cast<float>();
      float f = sampling_params.attr("frequency_penalty").cast<float>();
      float r = sampling_params.attr("repetition_penalty").cast<float>();
      float top_p = sampling_params.attr("top_p").cast<float>();
      float min_p = sampling_params.attr("min_p").cast<float>();
      auto seed = sampling_params.attr("seed").cast<std::optional<int64_t>>();

      bool is_greedy =
          py::int_(sampling_params.attr("sampling_type")).cast<int>() ==
          0;  // 0 corresponds to GREEDY

      int64_t top_k =
          std::min(sampling_params.attr("top_k").cast<int64_t>(), vocab_size);
      if (top_k == -1) {
        top_k = vocab_size;
      }

      if (temperature < _SAMPLING_EPS) {
        temperature = 1.0;
      }
      if (!do_top_p_top_k &&
          (top_p < 1.0 - _SAMPLING_EPS || top_k != vocab_size)) {
        do_top_p_top_k = true;
      }
      if (!do_min_p && min_p > _SAMPLING_EPS) {
        do_min_p = true;
      }
      if (!do_penalties &&
          (std::abs(p) >= _SAMPLING_EPS || std::abs(f) >= _SAMPLING_EPS ||
           std::abs(r - 1.0) >= _SAMPLING_EPS)) {
        do_penalties = true;
      }

      auto is_prompt = seq_group.attr("is_prompt").cast<bool>();
      auto prompt_logprobs = sampling_params.attr("prompt_logprobs");
      if (is_prompt && !prompt_logprobs.is_none()) {
        assert(!seq_group.attr("query_len").is_none());
        auto prefill_len =
            seq_group.attr("prompt_logprob_indices").cast<py::list>().size();
        temperatures.resize(temperatures.size() + prefill_len, temperature);
        top_ps.resize(top_ps.size() + prefill_len, top_p);
        top_ks.resize(top_ks.size() + prefill_len, top_k);
        min_ps.resize(min_ps.size() + prefill_len, min_p);
        presence_penalties.resize(presence_penalties.size() + prefill_len, 0);
        frequency_penalties.resize(frequency_penalties.size() + prefill_len, 0);
        repetition_penalties.resize(repetition_penalties.size() + prefill_len,
                                    1);
      }

      if (seq_group.attr("do_sample").cast<bool>()) {
        size_t sample_lens =
            seq_group.attr("sample_indices").cast<py::list>().size();
        assert(sample_lens > seq_ids.size());
        temperatures.resize(temperatures.size() + seq_ids.size(), temperature);
        top_ps.resize(top_ps.size() + seq_ids.size(), top_p);
        top_ks.resize(top_ks.size() + seq_ids.size(), top_k);
        min_ps.resize(min_ps.size() + seq_ids.size(), min_p);
        presence_penalties.resize(presence_penalties.size() + seq_ids.size(),
                                  p);
        frequency_penalties.resize(frequency_penalties.size() + seq_ids.size(),
                                   f);
        repetition_penalties.resize(
            repetition_penalties.size() + seq_ids.size(), r);
      }

      if (is_prompt) {
        prompt_best_of.push_back(sampling_params.attr("best_of").cast<int>());
      }

      for (auto seq_id : seq_ids) {
        auto seq_data = seq_group.attr("seq_data")[py::cast(seq_id)];
        auto final_entropy =
            std::vector<py::object>{seq_data.attr("get_len")()};
        if (extra_entropy.has_value()) {
          final_entropy.insert(final_entropy.end(),
                               extra_entropy.value().begin(),
                               extra_entropy.value().end());
        }
        final_entropy.push_back(py::int_(seq_id));
        auto seq_seeds = SamplingTensors::_get_sequence_seeds(
            seed, py::args(py::cast(final_entropy)), seeds_to_generate,
            is_greedy);
        sampling_seeds.insert(sampling_seeds.end(), seq_seeds.begin(),
                              seq_seeds.end());
      }

      auto sample_indices =
          seq_group.attr("sample_indices").cast<std::vector<int64_t>>();
      sample_indices.insert(sample_indices.end(), sample_indices.begin(),
                            sample_indices.end());
    }

    if (do_penalties) {
      for (const auto& seq_group : seq_groups) {
        auto sampling_params = seq_group.attr("sampling_params");
        auto prompt_logprobs = sampling_params.attr("prompt_logprobs");
        auto is_prompt = seq_group.attr("is_prompt").cast<bool>();
        auto do_sample = seq_group.attr("do_sample").cast<bool>();

        if (is_prompt && !prompt_logprobs.is_none()) {
          auto prefill_len = py::len(seq_group.attr("prompt_logprob_indices"));
          prompt_tokens.resize(prompt_tokens.size() + prefill_len);
          output_tokens.resize(output_tokens.size() + prefill_len);
        }

        if (do_sample) {
          auto seq_ids = seq_group.attr("seq_ids").cast<py::list>();
          py::dict seq_data = seq_group.attr("seq_data");
          for (auto seq_id : seq_ids) {
            auto seq_data_item = seq_data[seq_id];
            prompt_tokens.push_back(seq_data_item.attr("prompt_token_ids_array")
                                        .cast<py::buffer>());
            output_tokens.push_back(seq_data_item.attr("output_token_ids_array")
                                        .cast<py::buffer>());
          }
        }
      }
    }

    auto sampling_tensors = SamplingTensors::from_lists(
        temperatures, top_ps, top_ks, min_ps, presence_penalties,
        frequency_penalties, repetition_penalties, sampling_seeds,
        sample_indices, prompt_tokens, output_tokens, vocab_size,
        extra_seeds_to_generate, device, dtype);

    return std::make_tuple(sampling_tensors, do_penalties, do_top_p_top_k,
                           do_min_p);
  }

  static SamplingTensors from_lists(
      const std::vector<float>& temperatures, const std::vector<float>& top_ps,
      const std::vector<int>& top_ks, const std::vector<float>& min_ps,
      const std::vector<float>& presence_penalties,
      const std::vector<float>& frequency_penalties,
      const std::vector<float>& repetition_penalties,
      const std::vector<int64_t>& sampling_seeds,
      const std::vector<int64_t>& sample_indices,
      const std::vector<py::buffer>& prompt_tokens,
      const std::vector<py::buffer>& output_tokens,
      int64_t vocab_size, int64_t extra_seeds_to_generate,
      const torch::Device& device, const torch::ScalarType& dtype) {
    bool do_penalties = !prompt_tokens.empty() || !output_tokens.empty();

    torch::Tensor prompt_tensor, output_tensor;

    // auto start = std::chrono::high_resolution_clock::now();
    if (do_penalties) {
      auto prompt_padded_tokens = pad_tokens(prompt_tokens, vocab_size);
      auto output_padded_tokens = pad_tokens(output_tokens, vocab_size);
      prompt_tensor =
          torch::utils::tensor_from_numpy(prompt_padded_tokens.ptr());
      output_tensor =
          torch::utils::tensor_from_numpy(output_padded_tokens.ptr());
      if (pin_memory) {
        prompt_tensor = prompt_tensor.pin_memory();
        output_tensor = output_tensor.pin_memory();
      }
    }
    // auto finish = std::chrono::high_resolution_clock::now();
    // py::print(
    //     std::chrono::duration_cast<std::chrono::milliseconds>(finish - start)
    //         .count());

    auto create_tensor = [&](const auto& vec, torch::ScalarType dtype) {
      auto options = torch::TensorOptions().dtype(dtype).device(torch::kCPU);
      if (pin_memory) {
        options = options.pinned_memory(true);
      }
      auto tensor = torch::tensor(vec, options);
      return tensor;
    };

    auto temperatures_t = create_tensor(temperatures, dtype);
    auto top_ps_t = create_tensor(top_ps, dtype);
    auto min_ps_t = create_tensor(min_ps, dtype);
    auto presence_penalties_t = create_tensor(presence_penalties, dtype);
    auto frequency_penalties_t = create_tensor(frequency_penalties, dtype);
    auto repetition_penalties_t = create_tensor(repetition_penalties, dtype);
    auto top_ks_t = create_tensor(top_ks, torch::kInt32);
    auto sample_indices_t = create_tensor(sample_indices, torch::kInt64);

    auto sampling_seeds_t =
        create_tensor(sampling_seeds, torch::kInt64).t().contiguous();

    int64_t num_base_seeds = sampling_seeds_t.size(0) - extra_seeds_to_generate;
    auto sampling_seeds_gpu = sampling_seeds_t.to(device, true);
    std::optional<torch::Tensor> extra_seeds_gpu = std::nullopt;
    if (num_base_seeds < sampling_seeds_t.size(0)) {
      extra_seeds_gpu = sampling_seeds_gpu.slice(0, num_base_seeds);
    }
    sampling_seeds_gpu = sampling_seeds_gpu.slice(0, 0, num_base_seeds);

    torch::Tensor prompt_tokens_gpu, output_tokens_gpu;
    if (do_penalties) {
      prompt_tokens_gpu = prompt_tensor.to(device, true);
      output_tokens_gpu = output_tensor.to(device, true);
    } else {
      prompt_tokens_gpu = torch::empty(
          {0}, torch::TensorOptions().device(device).dtype(torch::kInt64));
      output_tokens_gpu = torch::empty(
          {0}, torch::TensorOptions().device(device).dtype(torch::kInt64));
    }

    return SamplingTensors{temperatures_t.to(device, true),
                           top_ps_t.to(device, true),
                           top_ks_t.to(device, true),
                           min_ps_t.to(device, true),
                           presence_penalties_t.to(device, true),
                           frequency_penalties_t.to(device, true),
                           repetition_penalties_t.to(device, true),
                           sampling_seeds_gpu,
                           sample_indices_t.to(device, true),
                           extra_seeds_gpu,
                           prompt_tokens_gpu,
                           output_tokens_gpu};
  }

  static std::vector<int64_t> _get_sequence_seeds(std::optional<int64_t> seed,
                                                  py::args extra_entropy,
                                                  int64_t seeds_to_generate,
                                                  bool is_greedy) {
    if (!is_greedy) {
      std::vector<int64_t> seq_seeds;
      seq_seeds.reserve(seeds_to_generate);

      std::mt19937_64 gen;
      if (!seed.has_value()) {
        static std::random_device rd;
        gen = std::mt19937_64{rd()};
      } else {
        std::string entropy_str = "(" + std::to_string(*seed);
        for (const auto& arg : extra_entropy) {
          entropy_str += ", " + std::to_string(arg.cast<int64_t>());
        }
        entropy_str += ")";
        std::seed_seq seed_seq(entropy_str.begin(), entropy_str.end());
        gen = std::mt19937_64{seed_seq};
      }
      std::uniform_int_distribution<int64_t> dist(
          std::numeric_limits<int64_t>::min(),
          std::numeric_limits<int64_t>::max());
      auto randint_fn = [&]() { return dist(gen); };

      for (int64_t i = 0; i < seeds_to_generate; ++i) {
        int64_t generated_seed = randint_fn();
        seq_seeds.push_back(generated_seed == 0 ? _SEED_0_REPLACEMENT
                                                : generated_seed);
      }

      return seq_seeds;
    } else {
      return std::vector<int64_t>(seeds_to_generate, 0);
    }
  }
};

PYBIND11_MODULE(_sampling_tensors, m) {
  auto utils = m.import("vllm.utils");
  auto is_pin_memory_available =
      utils.attr("is_pin_memory_available")().cast<bool>();
  pin_memory = is_pin_memory_available;
  
  py::class_<SamplingTensors>(m, "SamplingTensors")
      .def(py::init<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor,
                    torch::Tensor, std::optional<torch::Tensor>, torch::Tensor,
                    torch::Tensor>())
      .def_readwrite("temperatures", &SamplingTensors::temperatures)
      .def_readwrite("top_ps", &SamplingTensors::top_ps)
      .def_readwrite("top_ks", &SamplingTensors::top_ks)
      .def_readwrite("min_ps", &SamplingTensors::min_ps)
      .def_readwrite("presence_penalties", &SamplingTensors::presence_penalties)
      .def_readwrite("frequency_penalties",
                     &SamplingTensors::frequency_penalties)
      .def_readwrite("repetition_penalties",
                     &SamplingTensors::repetition_penalties)
      .def_readwrite("sampling_seeds", &SamplingTensors::sampling_seeds)
      .def_readwrite("sample_indices", &SamplingTensors::sample_indices)
      .def_readwrite("extra_seeds", &SamplingTensors::extra_seeds)
      .def_readwrite("prompt_tokens", &SamplingTensors::prompt_tokens)
      .def_readwrite("output_tokens", &SamplingTensors::output_tokens)
      .def_static("_get_sequence_seeds", &SamplingTensors::_get_sequence_seeds,
                  py::arg("seed"),  // py::arg("extra_entropy"),
                  py::arg("seeds_to_generate"), py::arg("is_greedy"))
      // .def_static("from_lists", &SamplingTensors::from_lists,
      //             py::arg("temperatures"), py::arg("top_ps"),
      //             py::arg("top_ks"), py::arg("min_ps"),
      //             py::arg("presence_penalties"),
      //             py::arg("frequency_penalties"),
      //             py::arg("repetition_penalties"), py::arg("sampling_seeds"),
      //             py::arg("sample_indices"), py::arg("prompt_tokens"),
      //             py::arg("output_tokens"), py::arg("vocab_size"),
      //             py::arg("extra_seeds_to_generate"), py::arg("device"),
      //             py::arg("dtype"))
      .def_static("test_numpy", [](py::array_t<int64_t> obj) { return obj; })
      .def_static("from_sampling_metadata",
                  &SamplingTensors::from_sampling_metadata,
                  py::arg("sampling_metadata"), py::arg("vocab_size"),
                  py::arg("device"), py::arg("dtype"), py::kw_only(),
                  py::arg("extra_seeds_to_generate") = 0,
                  py::arg("extra_entropy") = std::nullopt);
}
