// 突触矩阵 - 管理神经元之间的连接和权重更新
module synapse_matrix #(
    parameter PRE_NEURONS = 500,     // 前层神经元数量
    parameter POST_NEURONS = 10,     // 后层神经元数量
    parameter WEIGHT_WIDTH = 8       // 权重位宽
)(
    input wire clk,
    input wire rst_n,
    input wire learning_enable,      // 学习使能
    input wire weight_update_enable, // 权重更新使能
    input wire [PRE_NEURONS-1:0] pre_spikes,  // 前层神经元脉冲
    input wire [POST_NEURONS-1:0] post_spikes, // 后层神经元脉冲
    output wire [15:0] post_currents [0:POST_NEURONS-1] // 后层神经元输入电流
);

    // 存储突触权重
    reg [WEIGHT_WIDTH-1:0] weights [0:POST_NEURONS-1][0:PRE_NEURONS-1];
    
    // STDP参数
    localparam TAU_P = 20;
    localparam TAU_D = 20;
    localparam A_PLUS = 5;
    localparam A_MINUS = 3;
    
    // 生成突触和权重更新逻辑
    genvar i, j;
    generate
        for (i = 0; i < POST_NEURONS; i = i + 1) begin : post_neurons
            // 累加输入电流
            reg [15:0] current_sum;
            
            always @(posedge clk or negedge rst_n) begin
                if (!rst_n) begin
                    current_sum <= 0;
                end else begin
                    current_sum <= 0; // 重置累加器
                    for (int pre_idx = 0; pre_idx < PRE_NEURONS; pre_idx = pre_idx + 1) begin
                        if (pre_spikes[pre_idx])
                            current_sum <= current_sum + weights[i][pre_idx];
                    end
                end
            end
            
            assign post_currents[i] = current_sum;
            
            // 每个突触的STDP更新
            for (j = 0; j < PRE_NEURONS; j = j + 1) begin : synapses
                // 简化版的STDP更新
                always @(posedge clk or negedge rst_n) begin
                    if (!rst_n) begin
                        weights[i][j] <= 8'd128; // 初始化为中间值
                    end else if (weight_update_enable && learning_enable) begin
                        if (pre_spikes[j] && post_spikes[i]) begin
                            // 同时发放，增强权重
                            if (weights[i][j] < 8'd250)
                                weights[i][j] <= weights[i][j] + 5;
                        end else if (pre_spikes[j] && !post_spikes[i]) begin
                            // 仅前神经元发放，减弱权重
                            if (weights[i][j] > 8'd5)
                                weights[i][j] <= weights[i][j] - 3;
                        end else if (!pre_spikes[j] && post_spikes[i]) begin
                            // 仅后神经元发放，增强权重
                            if (weights[i][j] < 8'd250)
                                weights[i][j] <= weights[i][j] + 2;
                        end
                    end
                end
            end
        end
    endgenerate
    
endmodule