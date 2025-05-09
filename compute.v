// SNN计算核心 - 整合所有计算密集型模块
module snn_computation_core #(
    parameter INPUT_SIZE = 3072,    // 32x32x3
    parameter HIDDEN_SIZE = 500,
    parameter OUTPUT_SIZE = 10,
    parameter TIME_STEPS = 100      // 仿真时间步数
)(
    input wire clk,
    input wire rst_n,
    input wire start_compute,       // 开始计算
    input wire [7:0] input_data [0:INPUT_SIZE-1], // 输入数据
    input wire learning_mode,       // 学习模式
    output reg [OUTPUT_SIZE-1:0] output_spikes,   // 输出层脉冲
    output reg compute_done         // 计算完成标志
);

    // 时间步计数器
    reg [7:0] time_step;
    wire time_step_pulse;
    
    // 内部信号
    wire [INPUT_SIZE-1:0] input_spikes;
    wire [15:0] hidden_currents [0:HIDDEN_SIZE-1];
    wire [HIDDEN_SIZE-1:0] hidden_spikes;
    wire [15:0] hidden_potentials [0:HIDDEN_SIZE-1];
    wire [15:0] output_currents [0:OUTPUT_SIZE-1];
    wire [15:0] output_potentials [0:OUTPUT_SIZE-1];
    
    // 时间步控制
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            time_step <= 0;
            compute_done <= 0;
        end else if (start_compute) begin
            if (time_step < TIME_STEPS) begin
                time_step <= time_step + 1;
                compute_done <= 0;
            end else begin
                time_step <= 0;
                compute_done <= 1;
            end
        end
    end
    
    assign time_step_pulse = (start_compute && time_step < TIME_STEPS);
    
    // 1. 输入脉冲编码器
    spike_encoder #(
        .INPUT_SIZE(INPUT_SIZE),
        .MAX_RATE(255),
        .TIME_STEPS(TIME_STEPS)
    ) encoder (
        .clk(clk),
        .rst_n(rst_n),
        .pixel_data(input_data),
        .pixel_valid(start_compute && time_step == 0),
        .time_step_pulse(time_step_pulse),
        .spike_out(input_spikes)
    );
    
    // 2. 输入到隐藏层的突触矩阵
    synapse_matrix #(
        .PRE_NEURONS(INPUT_SIZE),
        .POST_NEURONS(HIDDEN_SIZE),
        .WEIGHT_WIDTH(8)
    ) input_hidden_synapses (
        .clk(clk),
        .rst_n(rst_n),
        .learning_enable(learning_mode),
        .weight_update_enable(time_step_pulse),
        .pre_spikes(input_spikes),
        .post_spikes(hidden_spikes),
        .post_currents(hidden_currents)
    );
    
    // 3. 隐藏层神经元
    genvar h;
    generate
        for (h = 0; h < HIDDEN_SIZE; h = h + 1) begin : hidden_neurons
            neuron_lif_improved #(
                .THRESHOLD(16'd550),
                .REST_POTENTIAL(16'd650),
                .RESET_POTENTIAL(16'd700),
                .TAU_M(8'd20),
                .REFRACTORY_PERIOD(4'd2)
            ) hidden_neuron (
                .clk(clk),
                .rst_n(rst_n),
                .enable(time_step_pulse),
                .input_current(hidden_currents[h]),
                .membrane_potential(hidden_potentials[h]),
                .spike(hidden_spikes[h])
            );
        end
    endgenerate
    
    // 4. 隐藏层到输出层的突触矩阵
    synapse_matrix #(
        .PRE_NEURONS(HIDDEN_SIZE),
        .POST_NEURONS(OUTPUT_SIZE),
        .WEIGHT_WIDTH(8)
    ) hidden_output_synapses (
        .clk(clk),
        .rst_n(rst_n),
        .learning_enable(learning_mode),
        .weight_update_enable(time_step_pulse),
        .pre_spikes(hidden_spikes),
        .post_spikes(output_spikes),
        .post_currents(output_currents)
    );
    
    // 5. 输出层神经元
    genvar o;
    generate
        for (o = 0; o < OUTPUT_SIZE; o = o + 1) begin : output_neurons
            neuron_lif_improved #(
                .THRESHOLD(16'd550),
                .REST_POTENTIAL(16'd650),
                .RESET_POTENTIAL(16'd700),
                .TAU_M(8'd20),
                .REFRACTORY_PERIOD(4'd2)
            ) output_neuron (
                .clk(clk),
                .rst_n(rst_n),
                .enable(time_step_pulse),
                .input_current(output_currents[o]),
                .membrane_potential(output_potentials[o]),
                .spike(output_spikes[o])
            );
        end
    endgenerate
    
endmodule