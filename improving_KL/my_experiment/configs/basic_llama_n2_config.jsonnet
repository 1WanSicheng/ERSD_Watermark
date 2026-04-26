// This is a jsonnet file for experiment config.
local UnImplementedError = function() {
  'error': error 'This is an abstract class, should be implemented',
};
// What is a experiment config?
// It is a json that contains the following fields:
local BaseConfig = {
  // data is stored at
  // data_root/${data_folder}/${task}/${to_display_model_name(model_str)}_${to_display_model_name(ref_model_str)}
  data_folder: UnImplementedError(),  // string
  // The name of the experiment.
  task: UnImplementedError(),  // string
  // model parameters
  model_str: UnImplementedError(),  // string
  ref_model_str: UnImplementedError(),  // string
  // dataset parameters
  ds_name: UnImplementedError(),  // string
  ds_cut_len: UnImplementedError(),  // int
  // worker parameters
  device: UnImplementedError(),  // string
  max_length: UnImplementedError(),  // int
  batch_size: UnImplementedError(),  // int
  private_key: UnImplementedError(),  // string
  methods: UnImplementedError(),  // list[string]
  ns: UnImplementedError(),  // list
  seeds: UnImplementedError(),  // list
  reweights: UnImplementedError(),  // list[string]
  print_output: UnImplementedError(),  // bool
  assert_cch: UnImplementedError(),  // bool
  assert_log_p_values: UnImplementedError(),  // bool
  // ray parameters
  repartition_size: UnImplementedError(),  // bool
};
local large_llamas = [
  'huggyllama/llama-7b',
  'huggyllama/llama-13b',
  'huggyllama/llama-65b',
  'daryl149/llama-2-7b-chat-hf',
  'daryl149/llama-2-13b-chat-hf',
  'daryl149/llama-2-70b-chat-hf',
];
local small_llamas = ['JackFram/llama-68m', 'JackFram/llama-160m'];

local large_opts = [
  'facebook/opt-6.7b',
  'facebook/opt-13b',
  'facebook/opt-30b',
  'facebook/opt-66b',
];
local small_opts = ['facebook/opt-1.3b', 'facebook/opt-125m', 'facebook/opt-350m'];

local large_gpts = [
  'openai-community/gpt2-large',
  'openai-community/gpt2-xl',
  'EleutherAI/gpt-neo-2.7B',
  'EleutherAI/gpt-neo-20B',
];
local small_gpts = ['gpt2', 'gpt2-medium'];

local large_gemmas = ['google/gemma-7b-it', 'google/gemma-7b'];
local small_gemmas = ['google/gemma-2b-it', 'google/gemma-2b'];

local large_qwen = ['Qwen/Qwen3-8B'];
local small_qwen = ['Qwen/Qwen3-0.6B'];

local DefaultConfig = BaseConfig {
  device: 'cuda:0',
  max_length: 250,
  batch_size: 4,
  private_key: '1234',
  print_output: false,
  methods: ['mc_uwm_synthid_psedo_r'],
  reweights: ['deltagumbel'],
  assert_cch: true,
  assert_log_p_values: true,
};
local Exp2_Verify_Config = DefaultConfig {
  data_folder: 'basic_sps',
  task: 'eli5',
  model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'eli5',
  ds_cut_len: 2000,
  ns: [2],
  seeds: [1],
  methods: ['mc'],
  reweights: [],
  repartition_size: 500,
  temperature: 1,
};
local configs = [
    Exp2_Verify_Config,
]
;
configs
