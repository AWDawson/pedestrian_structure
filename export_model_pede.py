# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import os
import sys
__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

from ppcls.modeling import architectures
from ppcls.utils.save_load import load_dygraph_pretrain
import paddle
import paddle.nn.functional as F
from paddle.jit import to_static

from model_export import *


def parse_args():
    def str2bool(v):
        return v.lower() in ("true", "t", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--backbone", type=str, default="resnet18_vd")
    parser.add_argument("-p", "--pretrained_model", type=str, default="./pede")
    parser.add_argument("-o", "--output_path", type=str, default="./inference_pede_r18")
    parser.add_argument("--load_static_weights", type=str2bool, default=False)
    #parser.add_argument("--img_size", type=int, default=256)

    return parser.parse_args()


def main():
    args = parse_args()

    model = CenterNet_ResNet(1, False, {'backbone': args.backbone})
    
    load_dygraph_pretrain(
        model,
        path=args.pretrained_model,
        load_static_weights=args.load_static_weights)
    model.eval()

    model = to_static(
        model,
        input_spec=[
            paddle.static.InputSpec(
                shape=[None, 3, 256, 128], dtype='float32')
        ])

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    paddle.jit.save(model, os.path.join(args.output_path, "inference"))


if __name__ == "__main__":
    main()

