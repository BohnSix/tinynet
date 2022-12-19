dic = {'conv_stem.weight': '_ef._conv_stem._conv._conv._conv.weight',
       'bn1.weight':'_ef._conv_stem._conv._bn.weight',
       'bn1.bias':'_ef._conv_stem._conv._bn.bias',
       'bn1.running_mean':'_ef._conv_stem._conv._bn._mean',
       'bn1.running_var':'_ef._conv_stem._conv._bn._variance',
       'blocks.0.0.conv_dw.weight':'_ef._blocks.0.._dcn._conv._conv._conv.weight',
       'blocks.0.0.bn1.weight':'_ef._blocks.0.._dcn._conv._bn.weight',
       'blocks.0.0.bn1.bias':'_ef._blocks.0.._dcn._conv._bn.bias',
        'blocks.0.0.bn1.running_mean':'_ef._blocks.0.._dcn._conv._bn._mean',
        'blocks.0.0.bn1.running_var':'_ef._blocks.0.._dcn._conv._bn._variance',
                'blocks.0.0.se.conv_reduce.weight':'_ef._blocks.0.._se._conv1._conv.weight',
                'blocks.0.0.se.conv_reduce.bias':'_ef._blocks.0.._se._conv1._conv.bias',
                'blocks.0.0.se.conv_expand.weight': '_ef._blocks.0.._se._conv2._conv.weight',
                'blocks.0.0.se.conv_expand.bias':'_ef._blocks.0.._se._conv2._conv.bias',
                'blocks.0.0.conv_pw.weight':'_ef._blocks.0.._pcn._conv._conv._conv.weight',
                'blocks.0.0.bn2.weight':'_ef._blocks.0.._pcn._conv._bn.weight',
                'blocks.0.0.bn2.bias':'_ef._blocks.0.._pcn._conv._bn.bias',
                'blocks.0.0.bn2.running_mean':'_ef._blocks.0.._pcn._conv._bn._mean',
                'blocks.0.0.bn2.running_var':'_ef._blocks.0.._pcn._conv._bn._variance',
                'blocks.1.0.conv_pw.weight':'_ef._blocks.1.._ecn._conv._conv._conv.weight',
                'blocks.1.0.bn1.weight':'_ef._blocks.1.._ecn._conv._bn.weight',
                'blocks.1.0.bn1.bias':'_ef._blocks.1.._ecn._conv._bn.bias',
                'blocks.1.0.bn1.running_mean':'_ef._blocks.1.._ecn._conv._bn._mean',
                'blocks.1.0.bn1.running_var':'_ef._blocks.1.._ecn._conv._bn._variance',
                'blocks.1.0.conv_dw.weight':'_ef._blocks.1.._dcn._conv._conv._conv.weight',
                'blocks.1.0.bn2.weight':'_ef._blocks.1.._dcn._conv._bn.weight',
                'blocks.1.0.bn2.bias':'_ef._blocks.1.._dcn._conv._bn.bias',
                'blocks.1.0.bn2.running_mean':'_ef._blocks.1.._dcn._conv._bn._mean',
                'blocks.1.0.bn2.running_var':'_ef._blocks.1.._dcn._conv._bn._variance',
                'blocks.1.0.se.conv_reduce.weight':'_ef._blocks.1.._se._conv1._conv.weight',
                'blocks.1.0.se.conv_reduce.bias':'_ef._blocks.1.._se._conv1._conv.bias',
                'blocks.1.0.se.conv_expand.weight':'_ef._blocks.1.._se._conv2._conv.weight',
                'blocks.1.0.se.conv_expand.bias':'_ef._blocks.1.._se._conv2._conv.bias',
                'blocks.1.0.conv_pwl.weight':'_ef._blocks.1.._pcn._conv._conv._conv.weight',
                'blocks.1.0.bn3.weight':'_ef._blocks.1.._pcn._conv._bn.weight',
                'blocks.1.0.bn3.bias':'_ef._blocks.1.._pcn._conv._bn.bias',
                'blocks.1.0.bn3.running_mean':'_ef._blocks.1.._pcn._conv._bn._mean',
                'blocks.1.0.bn3.running_var':'_ef._blocks.1.._pcn._conv._bn._variance',
                'blocks.1.1.conv_pw.weight':'_ef.block.2.._ecn._conv._conv._conv.weight',
                'blocks.1.1.bn1.weight':'_ef.block.2.._ecn._conv._bn.weight',
                'blocks.1.1.bn1.bias':'_ef.block.2.._ecn._conv._bn.bias',
                'blocks.1.1.bn1.running_mean':'_ef.block.2.._ecn._conv._bn._mean',
                'blocks.1.1.bn1.running_var':'_ef.block.2.._ecn._conv._bn._variance',
                'blocks.1.1.conv_dw.weight':'_ef.block.2.._dcn._conv._conv._conv.weight',
                'blocks.1.1.bn2.weight':'_ef.block.2.._dcn._conv._bn.weight',
                'blocks.1.1.bn2.bias':'_ef.block.2.._dcn._conv._bn.bias',
                'blocks.1.1.bn2.running_mean':'_ef.block.2.._dcn._conv._bn._mean',
                'blocks.1.1.bn2.running_var':'_ef.block.2.._dcn._conv._bn._variance',
                'blocks.1.1.se.conv_reduce.weight':'_ef.block.2.._se._conv1._conv.weight',
                'blocks.1.1.se.conv_reduce.bias':'_ef.block.2.._se._conv1._conv.bias',
                'blocks.1.1.se.conv_expand.weight':'_ef.block.2.._se._conv2._conv.weight',
                'blocks.1.1.se.conv_expand.bias':'_ef.block.2.._se._conv2._conv.bias',
                'blocks.1.1.conv_pwl.weight':'_ef.block.2.._pcn._conv._conv._conv.weight',
                'blocks.1.1.bn3.weight':'_ef.block.2.._pcn._conv._bn.weight',
                'blocks.1.1.bn3.bias':'_ef.block.2.._pcn._conv._bn.bias',
                'blocks.1.1.bn3.running_mean':'_ef.block.2.._pcn._conv._bn._mean',
                'blocks.1.1.bn3.running_var':'_ef.block.2.._pcn._conv._bn._variance',
                'blocks.2.0.conv_pw.weight':'_ef._blocks.3.._ecn._conv._conv._conv.weight',
                'blocks.2.0.bn1.weight':'_ef._blocks.3.._ecn._conv._bn.weight',
                'blocks.2.0.bn1.bias':'_ef._blocks.3.._ecn._conv._bn.bias',
                'blocks.2.0.bn1.running_mean':'_ef._blocks.3.._ecn._conv._bn._mean',
                'blocks.2.0.bn1.running_var':'_ef._blocks.3.._ecn._conv._bn._variance',
                'blocks.2.0.conv_dw.weight':'_ef._blocks.3.._dcn._conv._conv._conv.weight',
                'blocks.2.0.bn2.weight':'_ef._blocks.3.._dcn._conv._bn.weight',
                'blocks.2.0.bn2.bias':'_ef._blocks.3.._dcn._conv._bn.bias',
                'blocks.2.0.bn2.running_mean':'_ef._blocks.3.._dcn._conv._bn._mean',
                'blocks.2.0.bn2.running_var':'_ef._blocks.3.._dcn._conv._bn._variance',
                'blocks.2.0.se.conv_reduce.weight':'_ef._blocks.3.._se._conv1._conv.weight',
                'blocks.2.0.se.conv_reduce.bias':'_ef._blocks.3.._se._conv1._conv.bias',
                'blocks.2.0.se.conv_expand.weight':'_ef._blocks.3.._se._conv2._conv.weight',
                'blocks.2.0.se.conv_expand.bias':'_ef._blocks.3.._se._conv2._conv.bias',
                'blocks.2.0.conv_pwl.weight':'_ef._blocks.3.._pcn._conv._conv._conv.weight',
                'blocks.2.0.bn3.weight':'_ef._blocks.3.._pcn._conv._bn.weight',
                'blocks.2.0.bn3.bias':'_ef._blocks.3.._pcn._conv._bn.bias',
                'blocks.2.0.bn3.running_mean':'_ef._blocks.3.._pcn._conv._bn._mean',
                'blocks.2.0.bn3.running_var':'_ef._blocks.3.._pcn._conv._bn._variance',
                'blocks.2.1.conv_pw.weight':'_ef.block.4.._ecn._conv._conv._conv.weight',
                'blocks.2.1.bn1.weight':'_ef.block.4.._ecn._conv._bn.weight',
                'blocks.2.1.bn1.bias':'_ef.block.4.._ecn._conv._bn.bias',
                'blocks.2.1.bn1.running_mean':'_ef.block.4.._ecn._conv._bn._mean',
                'blocks.2.1.bn1.running_var':'_ef.block.4.._ecn._conv._bn._variance',
                'blocks.2.1.conv_dw.weight':'_ef.block.4.._dcn._conv._conv._conv.weight',
                'blocks.2.1.bn2.weight':'_ef.block.4.._dcn._conv._bn.weight',
                'blocks.2.1.bn2.bias':'_ef.block.4.._dcn._conv._bn.bias',
                'blocks.2.1.bn2.running_mean':'_ef.block.4.._dcn._conv._bn._mean',
                'blocks.2.1.bn2.running_var':'_ef.block.4.._dcn._conv._bn._variance',
                'blocks.2.1.se.conv_reduce.weight':'_ef.block.4.._se._conv1._conv.weight',
                'blocks.2.1.se.conv_reduce.bias':'_ef.block.4.._se._conv1._conv.bias',
                'blocks.2.1.se.conv_expand.weight':'_ef.block.4.._se._conv2._conv.weight',
                'blocks.2.1.se.conv_expand.bias':'_ef.block.4.._se._conv2._conv.bias',
                'blocks.2.1.conv_pwl.weight':'_ef.block.4.._pcn._conv._conv._conv.weight',
                'blocks.2.1.bn3.weight':'_ef.block.4.._pcn._conv._bn.weight',
                'blocks.2.1.bn3.bias':'_ef.block.4.._pcn._conv._bn.bias',
                'blocks.2.1.bn3.running_mean':'_ef.block.4.._pcn._conv._bn._mean',
                'blocks.2.1.bn3.running_var':'_ef.block.4.._pcn._conv._bn._variance',
                'blocks.3.0.conv_pw.weight':'_ef._blocks.5.._ecn._conv._conv._conv.weight',
                'blocks.3.0.bn1.weight':'_ef._blocks.5.._ecn._conv._bn.weight',
                'blocks.3.0.bn1.bias':'_ef._blocks.5.._ecn._conv._bn.bias',
                'blocks.3.0.bn1.running_mean':'_ef._blocks.5.._ecn._conv._bn._mean',
                'blocks.3.0.bn1.running_var':'_ef._blocks.5.._ecn._conv._bn._variance',
                'blocks.3.0.conv_dw.weight':'_ef._blocks.5.._dcn._conv._conv._conv.weight',
                'blocks.3.0.bn2.weight':'_ef._blocks.5.._dcn._conv._bn.weight',
                'blocks.3.0.bn2.bias':'_ef._blocks.5.._dcn._conv._bn.bias',
                'blocks.3.0.bn2.running_mean':'_ef._blocks.5.._dcn._conv._bn._mean',
                'blocks.3.0.bn2.running_var':'_ef._blocks.5.._dcn._conv._bn._variance',
                'blocks.3.0.se.conv_reduce.weight':'_ef._blocks.5.._se._conv1._conv.weight',
                'blocks.3.0.se.conv_reduce.bias':'_ef._blocks.5.._se._conv1._conv.bias',
                'blocks.3.0.se.conv_expand.weight':'_ef._blocks.5.._se._conv2._conv.weight',
                'blocks.3.0.se.conv_expand.bias':'_ef._blocks.5.._se._conv2._conv.bias',
                'blocks.3.0.conv_pwl.weight':'_ef._blocks.5.._pcn._conv._conv._conv.weight',
                'blocks.3.0.bn3.weight':'_ef._blocks.5.._pcn._conv._bn.weight',
                'blocks.3.0.bn3.bias':'_ef._blocks.5.._pcn._conv._bn.bias',
                'blocks.3.0.bn3.running_mean':'_ef._blocks.5.._pcn._conv._bn._mean',
                'blocks.3.0.bn3.running_var':'_ef._blocks.5.._pcn._conv._bn._variance',
                'blocks.3.1.conv_pw.weight':'_ef.block.6.._ecn._conv._conv._conv.weight',
                'blocks.3.1.bn1.weight':'_ef.block.6.._ecn._conv._bn.weight',
                'blocks.3.1.bn1.bias':'_ef.block.6.._ecn._conv._bn.bias',
                'blocks.3.1.bn1.running_mean':'_ef.block.6.._ecn._conv._bn._mean',
                'blocks.3.1.bn1.running_var':'_ef.block.6.._ecn._conv._bn._variance',
                'blocks.3.1.conv_dw.weight':'_ef.block.6.._dcn._conv._conv._conv.weight',
                'blocks.3.1.bn2.weight':'_ef.block.6.._dcn._conv._bn.weight',
                'blocks.3.1.bn2.bias':'_ef.block.6.._dcn._conv._bn.bias',
                'blocks.3.1.bn2.running_mean':'_ef.block.6.._dcn._conv._bn._mean',
                'blocks.3.1.bn2.running_var':'_ef.block.6.._dcn._conv._bn._variance',
                'blocks.3.1.se.conv_reduce.weight':'_ef.block.6.._se._conv1._conv.weight',
                'blocks.3.1.se.conv_reduce.bias':'_ef.block.6.._se._conv1._conv.bias',
                'blocks.3.1.se.conv_expand.weight':'_ef.block.6.._se._conv2._conv.weight',
                'blocks.3.1.se.conv_expand.bias':'_ef.block.6.._se._conv2._conv.bias',
                'blocks.3.1.conv_pwl.weight':'_ef.block.6.._pcn._conv._conv._conv.weight',
                'blocks.3.1.bn3.weight':'_ef.block.6.._pcn._conv._bn.weight',
                'blocks.3.1.bn3.bias':'_ef.block.6.._pcn._conv._bn.bias',
                'blocks.3.1.bn3.running_mean':'_ef.block.6.._pcn._conv._bn._mean',
                'blocks.3.1.bn3.running_var':'_ef.block.6.._pcn._conv._bn._variance',
                'blocks.3.2.conv_pw.weight':'_ef._blocks.7.._ecn._conv._conv._conv.weight',
                'blocks.3.2.bn1.weight':'_ef._blocks.7.._ecn._conv._bn.weight',
                'blocks.3.2.bn1.bias':'_ef._blocks.7.._ecn._conv._bn.bias',
                'blocks.3.2.bn1.running_mean':'_ef._blocks.7.._ecn._conv._bn._mean',
                'blocks.3.2.bn1.running_var':'_ef._blocks.7.._ecn._conv._bn._variance',
                'blocks.3.2.conv_dw.weight':'_ef._blocks.7.._dcn._conv._conv._conv.weight',
                'blocks.3.2.bn2.weight':'_ef._blocks.7.._dcn._conv._bn.weight',
                'blocks.3.2.bn2.bias':'_ef._blocks.7.._dcn._conv._bn.bias',
                'blocks.3.2.bn2.running_mean':'_ef._blocks.7.._dcn._conv._bn._mean',
                'blocks.3.2.bn2.running_var':'_ef._blocks.7.._dcn._conv._bn._variance',
                'blocks.3.2.se.conv_reduce.weight':'_ef._blocks.7.._se._conv1._conv.weight',
                'blocks.3.2.se.conv_reduce.bias':'_ef._blocks.7.._se._conv1._conv.bias',
                'blocks.3.2.se.conv_expand.weight':'_ef._blocks.7.._se._conv2._conv.weight',
                'blocks.3.2.se.conv_expand.bias':'_ef._blocks.7.._se._conv2._conv.bias',
                'blocks.3.2.conv_pwl.weight':'_ef._blocks.7.._pcn._conv._conv._conv.weight',
                'blocks.3.2.bn3.weight':'_ef._blocks.7.._pcn._conv._bn.weight',
                'blocks.3.2.bn3.bias':'_ef._blocks.7.._pcn._conv._bn.bias',
                'blocks.3.2.bn3.running_mean':'_ef._blocks.7.._pcn._conv._bn._mean',
                'blocks.3.2.bn3.running_var':'_ef._blocks.7.._pcn._conv._bn._variance',
                'blocks.4.0.conv_pw.weight':'_ef.block.8.._ecn._conv._conv._conv.weight',
                'blocks.4.0.bn1.weight':'_ef.block.8.._ecn._conv._bn.weight',
                'blocks.4.0.bn1.bias':'_ef.block.8.._ecn._conv._bn.bias',
                'blocks.4.0.bn1.running_mean':'_ef.block.8.._ecn._conv._bn._mean',
                'blocks.4.0.bn1.running_var':'_ef.block.8.._ecn._conv._bn._variance',
                'blocks.4.0.conv_dw.weight':'_ef.block.8.._dcn._conv._conv._conv.weight',
                'blocks.4.0.bn2.weight':'_ef.block.8.._dcn._conv._bn.weight',
                'blocks.4.0.bn2.bias':'_ef.block.8.._dcn._conv._bn.bias',
                'blocks.4.0.bn2.running_mean':'_ef.block.8.._dcn._conv._bn._mean',
                'blocks.4.0.bn2.running_var':'_ef.block.8.._dcn._conv._bn._variance',
                'blocks.4.0.se.conv_reduce.weight':'_ef.block.8.._se._conv1._conv.weight',
                'blocks.4.0.se.conv_reduce.bias':'_ef.block.8.._se._conv1._conv.bias',
                'blocks.4.0.se.conv_expand.weight':'_ef.block.8.._se._conv2._conv.weight',
                'blocks.4.0.se.conv_expand.bias':'_ef.block.8.._se._conv2._conv.bias',
                'blocks.4.0.conv_pwl.weight':'_ef.block.8.._pcn._conv._conv._conv.weight',
                'blocks.4.0.bn3.weight':'_ef.block.8.._pcn._conv._bn.weight',
                'blocks.4.0.bn3.bias':'_ef.block.8.._pcn._conv._bn.bias',
                'blocks.4.0.bn3.running_mean':'_ef.block.8.._pcn._conv._bn._mean',
                'blocks.4.0.bn3.running_var':'_ef.block.8.._pcn._conv._bn._variance',
                'blocks.4.1.conv_pw.weight':'_ef._blocks.9.._ecn._conv._conv._conv.weight',
                'blocks.4.1.bn1.weight':'_ef._blocks.9.._ecn._conv._bn.weight',
                'blocks.4.1.bn1.bias':'_ef._blocks.9.._ecn._conv._bn.bias',
                'blocks.4.1.bn1.running_mean':'_ef._blocks.9.._ecn._conv._bn._mean',
                'blocks.4.1.bn1.running_var':'_ef._bloc ks.9.._ecn._conv._bn._variance',
                'blocks.4.1.conv_dw.weight':'_ef._blocks.9.._dcn._conv._conv._conv.weight',
                'blocks.4.1.bn2.weight':'_ef._blocks.9.._dcn._conv._bn.weight',
                'blocks.4.1.bn2.bias':'_ef._blocks.9.._dcn._conv._bn.bias',
                'blocks.4.1.bn2.running_mean':'_ef._blocks.9.._dcn._conv._bn._mean',
                'blocks.4.1.bn2.running_var':'_ef._blocks.9.._dcn._conv._bn._variance',
                'blocks.4.1.se.conv_reduce.weight':'_ef._blocks.9.._se._conv1._conv.weight',
                'blocks.4.1.se.conv_reduce.bias':'_ef._blocks.9.._se._conv1._conv.bias',
                'blocks.4.1.se.conv_expand.weight':'_ef._blocks.9.._se._conv2._conv.weight',
                'blocks.4.1.se.conv_expand.bias':'_ef._blocks.9.._se._conv2._conv.bias',
                'blocks.4.1.conv_pwl.weight':'_ef._blocks.9.._pcn._conv._conv._conv.weight',
                'blocks.4.1.bn3.weight':'_ef._blocks.9.._pcn._conv._bn.weight',
                'blocks.4.1.bn3.bias':'_ef._blocks.9.._pcn._conv._bn.bias',
                'blocks.4.1.bn3.running_mean':'_ef._blocks.9.._pcn._conv._bn._mean',
                'blocks.4.1.bn3.running_var':'_ef._blocks.9.._pcn._conv._bn._variance',
                'blocks.4.2.conv_pw.weight':'_ef.block.10.._ecn._conv._conv._conv.weight',
                'blocks.4.2.bn1.weight':'_ef.block.10.._ecn._conv._bn.weight',
                'blocks.4.2.bn1.bias':'_ef.block.10.._ecn._conv._bn.bias',
                'blocks.4.2.bn1.running_mean':'_ef.block.10.._ecn._conv._bn._mean',
                'blocks.4.2.bn1.running_var':'_ef.block.10.._ecn._conv._bn._variance',
                'blocks.4.2.conv_dw.weight':'_ef.block.10.._dcn._conv._conv._conv.weight',
                'blocks.4.2.bn2.weight':'_ef.block.10.._dcn._conv._bn.weight',
                'blocks.4.2.bn2.bias':'_ef.block.10.._dcn._conv._bn.bias',
                'blocks.4.2.bn2.running_mean':'_ef.block.10.._dcn._conv._bn._mean',
                'blocks.4.2.bn2.running_var':'_ef.block.10.._dcn._conv._bn._variance',
                'blocks.4.2.se.conv_reduce.weight':'_ef.block.10.._se._conv1._conv.weight',
                'blocks.4.2.se.conv_reduce.bias':'_ef.block.10.._se._conv1._conv.bias',
                'blocks.4.2.se.conv_expand.weight':'_ef.block.10.._se._conv2._conv.weight',
                'blocks.4.2.se.conv_expand.bias':'_ef.block.10.._se._conv2._conv.bias',
                'blocks.4.2.conv_pwl.weight':'_ef.block.10.._pcn._conv._conv._conv.weight',
                'blocks.4.2.bn3.weight':'_ef.block.10.._pcn._conv._bn.weight',
                'blocks.4.2.bn3.bias':'_ef.block.10.._pcn._conv._bn.bias',
                'blocks.4.2.bn3.running_mean':'_ef.block.10.._pcn._conv._bn._mean',
                'blocks.4.2.bn3.running_var':'_ef.block.10.._pcn._conv._bn._variance',
                'blocks.5.0.conv_pw.weight':'_ef.block.11.._ecn._conv._conv._conv.weight',
                'blocks.5.0.bn1.weight':'_ef.block.11.._ecn._conv._bn.weight',
                'blocks.5.0.bn1.bias':'_ef.block.11.._ecn._conv._bn.bias',
                'blocks.5.0.bn1.running_mean':'_ef.block.11.._ecn._conv._bn._mean',
                'blocks.5.0.bn1.running_var':'_ef.block.11.._ecn._conv._bn._variance',
                'blocks.5.0.conv_dw.weight':'_ef.block.11.._dcn._conv._conv._conv.weight',
                'blocks.5.0.bn2.weight':'_ef.block.11.._dcn._conv._bn.weight',
                'blocks.5.0.bn2.bias':'_ef.block.11.._dcn._conv._bn.bias',
                'blocks.5.0.bn2.running_mean':'_ef.block.11.._dcn._conv._bn._mean',
                'blocks.5.0.bn2.running_var':'_ef.block.11.._dcn._conv._bn._variance',
                'blocks.5.0.se.conv_reduce.weight':'_ef.block.11.._se._conv1._conv.weight',
                'blocks.5.0.se.conv_reduce.bias':'_ef.block.11.._se._conv1._conv.bias',
                'blocks.5.0.se.conv_expand.weight':'_ef.block.11.._se._conv2._conv.weight',
                'blocks.5.0.se.conv_expand.bias':'_ef.block.11.._se._conv2._conv.bias',
                'blocks.5.0.conv_pwl.weight':'_ef.block.11.._pcn._conv._conv._conv.weight',
                'blocks.5.0.bn3.weight':'_ef.block.11.._pcn._conv._bn.weight',
                'blocks.5.0.bn3.bias':'_ef.block.11.._pcn._conv._bn.bias',
                'blocks.5.0.bn3.running_mean':'_ef.block.11.._pcn._conv._bn._mean',
                'blocks.5.0.bn3.running_var':'_ef.block.11.._pcn._conv._bn._variance',
                'blocks.5.1.conv_pw.weight':'_ef._blocks.12.._ecn._conv._conv._conv.weight',
                'blocks.5.1.bn1.weight':'_ef._blocks.12.._ecn._conv._bn.weight',
                'blocks.5.1.bn1.bias':'_ef._blocks.12.._ecn._conv._bn.bias',
                'blocks.5.1.bn1.running_mean':'_ef._blocks.12.._ecn._conv._bn._mean',
                'blocks.5.1.bn1.running_var':'_ef._blocks.12.._ecn._conv._bn._variance',
                'blocks.5.1.conv_dw.weight':'_ef._blocks.12.._dcn._conv._conv._conv.weight',
                'blocks.5.1.bn2.weight':'_ef._blocks.12.._dcn._conv._bn.weight',
                'blocks.5.1.bn2.bias':'_ef._blocks.12.._dcn._conv._bn.bias',
                'blocks.5.1.bn2.running_mean':'_ef._blocks.12.._dcn._conv._bn._mean',
                'blocks.5.1.bn2.running_var':'_ef._blocks.12.._dcn._conv._bn._variance',
                'blocks.5.1.se.conv_reduce.weight':'_ef._blocks.12.._se._conv1._conv.weight',
                'blocks.5.1.se.conv_reduce.bias':'_ef._blocks.12.._se._conv1._conv.bias',
                'blocks.5.1.se.conv_expand.weight':'_ef._blocks.12.._se._conv2._conv.weight',
                'blocks.5.1.se.conv_expand.bias':'_ef._blocks.12.._se._conv2._conv.bias',
                'blocks.5.1.conv_pwl.weight':'_ef._blocks.12.._pcn._conv._conv._conv.weight',
                'blocks.5.1.bn3.weight':'_ef._blocks.12.._pcn._conv._bn.weight',
                'blocks.5.1.bn3.bias':'_ef._blocks.12.._pcn._conv._bn.bias',
                'blocks.5.1.bn3.running_mean':'_ef._blocks.12.._pcn._conv._bn._mean',
                'blocks.5.1.bn3.running_var':'_ef._blocks.12.._pcn._conv._bn._variance',
                # 'blocks.5.2.conv_pw.weight:
                # 'blocks.5.2.bn1.weight
                # 'blocks.5.2.bn1.bias
                # 'blocks.5.2.bn1.running_mean
                # 'blocks.5.2.bn1.running_var
                # 'blocks.5.2.conv_dw.weight
                # 'blocks.5.2.bn2.weight
                # 'blocks.5.2.bn2.bias
                # 'blocks.5.2.bn2.running_mean
                # 'blocks.5.2.bn2.running_var
                # 'blocks.5.2.se.conv_reduce.weight
                # 'blocks.5.2.se.conv_reduce.bias
                # 'blocks.5.2.se.conv_expand.weight
                # 'blocks.5.2.se.conv_expand.bias
                # 'blocks.5.2.conv_pwl.weight
                # 'blocks.5.2.bn3.weight
                # 'blocks.5.2.bn3.bias
                # 'blocks.5.2.bn3.running_mean
                # 'blocks.5.2.bn3.running_var
                # 'blocks.5.3.conv_pw.weight
                # 'blocks.5.3.bn1.weight
                # 'blocks.5.3.bn1.bias
                # 'blocks.5.3.bn1.running_mean
                # 'blocks.5.3.bn1.running_var
                # 'blocks.5.3.conv_dw.weight
                # 'blocks.5.3.bn2.weight
                # 'blocks.5.3.bn2.bias
                # 'blocks.5.3.bn2.running_mean
                # 'blocks.5.3.bn2.running_var
                # 'blocks.5.3.se.conv_reduce.weight
                # 'blocks.5.3.se.conv_reduce.bias
                # 'blocks.5.3.se.conv_expand.weight
                # 'blocks.5.3.se.conv_expand.bias
                # 'blocks.5.3.conv_pwl.weight
                # 'blocks.5.3.bn3.weight
                # 'blocks.5.3.bn3.bias
                # 'blocks.5.3.bn3.running_mean
                # 'blocks.5.3.bn3.running_var
                # 'blocks.6.0.conv_pw.weight
                # 'blocks.6.0.bn1.weight
                # 'blocks.6.0.bn1.bias
                # 'blocks.6.0.bn1.running_mean
                # 'blocks.6.0.bn1.running_var
                # 'blocks.6.0.conv_dw.weight
                # 'blocks.6.0.bn2.weight
                # 'blocks.6.0.bn2.bias
                # 'blocks.6.0.bn2.running_mean
                # 'blocks.6.0.bn2.running_var
                # 'blocks.6.0.se.conv_reduce.weight
                # 'blocks.6.0.se.conv_reduce.bias
                # 'blocks.6.0.se.conv_expand.weight
                # 'blocks.6.0.se.conv_expand.bias
                # 'blocks.6.0.conv_pwl.weight
                # 'blocks.6.0.bn3.weight
                # 'blocks.6.0.bn3.bias
                # 'blocks.6.0.bn3.running_mean
                # 'blocks.6.0.bn3.running_var
                'conv_head.weight':' _conv._conv._conv.weight',
                'bn2.weight':'_conv._bn.weight',
                'bn2.bias':'_conv._bn.bias',
                'bn2.running_mean':'_conv._bn._mean',
                'bn2.running_var':'_conv._bn._variance',
                'classifier.weight':'_fc.weight',
                'classifier.bias':'_fc.bias'
}
