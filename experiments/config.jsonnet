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
  temperature: UnImplementedError(),  // number
  top_k: UnImplementedError(),  // int
  top_p: UnImplementedError(),  // number
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
  '/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/model-weights/huggyllama__llama-7b',
];
local small_llamas = [
  '/mnt/workspace0/A24738/AcceleratedUnbiasedWatermark-main/model-weights/JackFram__llama-68m',
];

local large_qwens = [
  '/mnt/workspace0/A24738/model-weights/Qwen2.5-7B-Instruct',
];
local small_qwens = [
  '/mnt/workspace0/A24738/model-weights/Qwen2.5-0.5B-Instruct',
];

local DefaultConfig = BaseConfig {
  device: 'cuda:0',
  max_length: 32,
  batch_size: 1,
  temperature: 1.0,
  top_k: 0,
  top_p: 1.0,
  private_key: '1234',
  print_output: false,
  methods: ['basic', 'basic_uwm', 'mc', 'mc_uwm_strength', 'mc_uwm_speed', 'ersd', 'ersd_wm'],
  reweights: ['deltagumbel', 'gamma'],
  assert_cch: true,
  assert_log_p_values: true,
};
local Debug_Config = DefaultConfig {
  data_folder: 'debug_data',
  task: 'summarization_scan_n',
  model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  ds_cut_len: 1,
  ns: [1],
  seeds: std.range(1, 3),
  // reweights: ['deltagumbel'],
  batch_size: 10000,
  repartition_size: 1,
  // methods: ['mc', 'mc_uwm_strength'],
  // methods: ['mc', 'mc_uwm_speed'],
  methods: ['mc', 'basic'],
  assert_cch: true,
  assert_log_p_values: true,
  print_output: true,
};
local Exp1_Verify_Config = DefaultConfig {
  data_folder: 'verify_data',
  task: 'summarization_scan_n',
  model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  ds_cut_len: 50,
  ns: [1],
  seeds: std.range(1, 10),
  repartition_size: 500,
};
local Exp2_Config_Template = DefaultConfig {
  data_folder: 'data',
  task: 'summarization_scan_n',
  // model_str: large_llamas[0],
  ref_model_str: small_llamas[0],
  ds_name: 'summarization',
  ds_cut_len: 1000,
  ns: [1, 2, 3, 4, 5, 6, 7, 8],
  top_k: 0,
  top_p: 1.0,
  methods: ['ersd', 'ersd_wm'],
  reweights: ['deltagumbel'],
  seeds: [1],
  repartition_size: 50,
};
local Exp3_Config_Template = Exp2_Config_Template {
  data_folder: 'data',
  task: 'oeg_scan_n',
  ds_name: 'oeg',
};
local Exp1_RaceSharpness_Config = DefaultConfig {
  data_folder: 'extra_exp1_data',
  task: 'summarization_scan_n',
  model_str: large_qwens[0],
  ref_model_str: small_qwens[0],
  ds_name: 'summarization',
  ds_cut_len: 200,
  ns: [1, 2, 3, 4, 5, 6, 7, 8],
  top_k: 0,
  top_p: 1.0,
  methods: ['ersd', 'ersd_wm', 'ersd_nocc', 'ersd_nocc_wm'],
  reweights: ['deltagumbel'],
  seeds: [1],
  repartition_size: 50,
};
local configs = [
                  // Debug_Config,
                  // Exp1_Verify_Config,
                ]
                + [
                  Exp2_Config_Template {
                    model_str: model_str,
                  }
                  for model_str in large_llamas
                ]
                + [
                  Exp1_RaceSharpness_Config
                ]
;
configs
