-- modified code based on the one from https://github.com/wojzaremba/lstm
-- add data interface for running the lstm on reber grammar test

require('cunn')
require('nngraph')
require('base')
local ptb = require('data')

deviceParams = cutorch.getDeviceProperties(1)
cudaComputeCapability = deviceParams.major + deviceParams.minor/10
LookupTable = nn.LookupTable

-- parameter setting
local params = {batch_size=50,
                seq_length=10, -- as max RG length is set at 10
                data_path="./rb.data.txt",
                decay=2,
                rnn_size=10,
                dropout=0,
                init_weight=0.1,
                lr=1,
                vocab_size=7,
                --max_epoch=4,
                max_max_epoch=250,
                max_grad_norm=5}

-- transfer data from memory to gpu
local function transfer_data(x)
    return x:cuda()
end

-- train/valid/test data
local state_train,state_valid,state_test
local model = {}
-- derivatives?
local paramx, paramdx

-- lstm step function, no peephole weight, modified output gate
local function lstm(x, prev_c, prev_h)
  -- Calculate all four gates in one go/preactivation
  local i2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(x)
  local h2h = nn.Linear(params.rnn_size, 4*params.rnn_size)(prev_h)
  local gates = nn.CAddTable()({i2h, h2h})
  
  -- Reshape to (batch_size, n_gates, hid_size)
  -- Then slize the n_gates dimension, i.e dimension 2
  local reshaped_gates =  nn.Reshape(4,params.rnn_size)(gates)
  local sliced_gates = nn.SplitTable(2)(reshaped_gates) -- split rows
  
  -- Use select gate to fetch each gate and apply nonlinearity
  local in_gate          = nn.Sigmoid()(nn.SelectTable(1)(sliced_gates))
  local in_transform     = nn.Tanh()(nn.SelectTable(2)(sliced_gates))
  local forget_gate      = nn.Sigmoid()(nn.SelectTable(3)(sliced_gates))
  local out_gate         = nn.Sigmoid()(nn.SelectTable(4)(sliced_gates))

  local next_c           = nn.CAddTable()({
      nn.CMulTable()({forget_gate, prev_c}),
      nn.CMulTable()({in_gate,     in_transform})
  })
  local next_h           = nn.CMulTable()({out_gate, nn.Tanh()(next_c)})

  return next_c, next_h
end

-- chain step function into complete network
local function create_network()
  -- Identity: creates a module that returns whatever is input to it as output
  -- input units
  local x                = nn.Identity()()
  -- meta output units
  local y                = nn.Identity()()
  -- recurrent output between lstm step function [c,h]
  local prev_s           = nn.Identity()()

  -- i is a dictionary, where the statement sets the i[0] value to be the lookuptable
  -- convolutional layer for string
  local i                = {[0] = LookupTable(params.vocab_size,
                                                    params.rnn_size)(x)}
  local next_s           = {}
  local split         = {prev_s:split(2)}
  
  local prev_c         = split[1]
  local prev_h         = split[2]
  local dropped        = nn.Dropout(params.dropout)(i[0])
  local next_c, next_h = lstm(dropped, prev_c, prev_h)

  -- form the new recurrent output
  table.insert(next_s, next_c)
  table.insert(next_s, next_h)

  i[1] = next_h
  
  local h2y              = nn.Linear(params.rnn_size, params.vocab_size)
  -- only perform dropout at lstm output units/ can choose to use/not use drop out
  local dropped          = nn.Dropout(params.dropout)(i[1])
  -- perform prediction at dropout layer
  local pred             = nn.LogSoftMax()(h2y(dropped))

  -- criterion
  local err              = nn.ClassNLLCriterion()({pred, y})

  -- conclude the module
  local module           = nn.gModule({x, y, prev_s},
                                      {err, nn.Identity()(next_s)})
  -- initialize parameter
  module:getParameters():uniform(-params.init_weight, params.init_weight)
  return transfer_data(module) -- return data transfered to GPU
end

-- setup the network to be ready
local function setup()
  print("Creating a LSTM network.")

  local core_network = create_network()
  -- getParameters() will return [flatparameters,flatGradParameters]
  paramx, paramdx = core_network:getParameters() 

  model.s = {} --state
  model.ds = {} --  state derivatives?
  model.start_s = {} -- starting state

  -- TODO, to adapt to the Reber grammar where sequences do not have identical length
  for j = 0, params.seq_length do
    model.s[j] = {}
    for d = 1, 2 do
      model.s[j][d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size)) -- To GPU
    end
  end

  for d = 1, 2 do
    model.start_s[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size)) -- To GPU
    model.ds[d] = transfer_data(torch.zeros(params.batch_size, params.rnn_size)) -- To GPU
  end

  model.core_network = core_network
  model.rnns = g_cloneManyTimes(core_network, params.seq_length) -- specialized function for unrolling the model
  model.norm_dw = 0
  model.err = transfer_data(torch.zeros(params.seq_length)) -- To GPU
end

-- reset state
local function reset_state(state)
  state.pos = 1
  if model ~= nil and model.start_s ~= nil then
    for d = 1, 2 do
      model.start_s[d]:zero()
    end
  end
end

-- reset state's derivatives
local function reset_ds()
  for d = 1, #model.ds do
    model.ds[d]:zero()
  end
end

-- forward propagation
local function fp(state)
  g_replace_table(model.s[0], model.start_s)
  if state.pos + params.seq_length > state.data:size(1) then
    reset_state(state)
  end

  for i = 1, params.seq_length do
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    model.err[i], model.s[i] = unpack(model.rnns[i]:forward({x, y, s}))
    state.pos = state.pos + 1
  end

  g_replace_table(model.start_s, model.s[params.seq_length]) -- specialized function for update parameter?

  return model.err:mean()
end

-- backward propagation
local function bp(state)

  paramdx:zero()
  reset_ds()

  for i = params.seq_length, 1, -1 do
    state.pos = state.pos - 1
    local x = state.data[state.pos]
    local y = state.data[state.pos + 1]
    local s = model.s[i - 1]
    local derr = transfer_data(torch.ones(1))
    local tmp = model.rnns[i]:backward({x, y, s},
                                       {derr, model.ds})[3]
    g_replace_table(model.ds, tmp)
    cutorch.synchronize()
  end

  state.pos = state.pos + params.seq_length
  model.norm_dw = paramdx:norm()

  if model.norm_dw > params.max_grad_norm then
    local shrink_factor = params.max_grad_norm / model.norm_dw
    paramdx:mul(shrink_factor)
  end

  paramx:add(paramdx:mul(-params.lr))
end

-- main function entry
local function main()
  g_init_gpu(arg) -- specialized function for initializing the gpu

  -- load data into GPU
  state_data = {data=transfer_data(ptb.loaddataset(params.data_path,params.batch_size))}
  
  print("Network parameters:")
  print(params)

  --local states = {state_train, state_valid, state_test}
  local states = {state_data}

  for _, state in pairs(states) do
    reset_state(state)
  end


  setup()
  local step = 0
  local epoch = 0
  local total_cases = 0
  local beginning_time = torch.tic()
  local start_time = torch.tic()
  print("Starting training.")
  local words_per_step = params.seq_length * params.batch_size -- number of words per step/mini-batch
  local epoch_size = torch.floor(state_data.data:size(1) / params.seq_length) -- number of mini-batches per epoch
  local perps
  while epoch < params.max_max_epoch do
    local perp = fp(state_data)
    if perps == nil then
      perps = torch.zeros(epoch_size):add(perp)
    end
    perps[step % epoch_size + 1] = perp
    step = step + 1
    bp(state_data)
    total_cases = total_cases + params.seq_length * params.batch_size
    
    epoch = step / epoch_size
    if step % 10 == 0 then -- print intermediate result for every 10 steps
      print('epoch = ' .. g_f3(epoch) .. ', train prep. = ' .. g_f3(torch.exp(perps:mean())) .. ',step = ' .. g_f3(step))
    end
    
    if step % 33 == 0 then
      cutorch.synchronize()
      collectgarbage()
    end
  end
  -- run_test()
  print("Training is over.")
  local since_beginning = g_d(torch.toc(beginning_time) / 60)
  print('train perp. = ' .. g_f3(torch.exp(perps:mean())) .. ',since beginning = ' .. since_beginning .. 'mins.')
end

main()
