# flash_attn_func v3 的 ATK 测试

这组文件测试的是 `MinghuasLab/flash-attention-npu` 的
`flash_attn_npu_v3.flash_attn_interface.flash_attn_func`，不是
`torch_npu.npu_fusion_attention`，也不是 ACLNN 接口。

> 注意：`flash_attn_npu 0.2.0b1` 的 `flash_attn_npu_v3/__init__.py` 只重新导出了
> `flash_attn_with_kvcache`，没有重新导出 `flash_attn_func`。因此不能使用
> `from flash_attn_npu_v3 import flash_attn_func`；之前版本的测试插件在这里有误，
> 当前文件已经改为从 `flash_attn_npu_v3.flash_attn_interface` 导入。

测试场景固定为：

- `q/k/v`: `[1, 128, 8, 64]`, `fp16`
- `causal=True`
- 预期 `out.shape`: `[1, 128, 8, 64]`
- NPU 节点调用真实扩展；CPU 节点计算 fp32 数学参考值

## 1. 前置条件

在 Atlas A2/A3（Ascend 910B/C）环境中安装 CANN、PyTorch/torch_npu、ATK，并按
`MinghuasLab/flash-attention-npu` 的说明编译安装 v3 扩展。必须使用运行 ATK 的
同一个 Python 环境安装：

```bash
which python
which atk
python -m pip show flash-attn-npu
python -c "from flash_attn_npu_v3.flash_attn_interface import flash_attn_func; print(flash_attn_func)"
python -c "import flash_attn_npu_3 as m; print(m.__file__); print(dir(m))"
```

如果尚未安装，在仓库根目录执行：

```bash
source /usr/local/Ascend/ascend-toolkit/latest/bin/setenv.bash
FLASH_ATTN_BUILD_VERSION=v3 python -m pip install -v .
```

如果服务器上没有仓库，可选以下任一路径。

联网环境从 PyPI 获取官方源码包：

```bash
mkdir -p /workspace/flash_attn_src
cd /workspace/flash_attn_src
python -m pip download --no-deps --no-binary=:all: flash-attn-npu==0.2.0b1
tar -xzf flash_attn_npu-0.2.0b1.tar.gz
cd flash_attn_npu-0.2.0b1
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  python -m pip install -v --no-build-isolation --no-deps .
```

离线环境：把修正版目录里的 `flash_attn_npu-0.2.0b1.tar.gz` 上传到服务器，
校验 SHA256 后安装：

```bash
sha256sum flash_attn_npu-0.2.0b1.tar.gz
# 期望：10660b8a043ecc018dc5ba439348d1ba4a5c789d472b69e890afd973320139ca

mkdir -p /workspace/flash_attn_src
tar -xzf flash_attn_npu-0.2.0b1.tar.gz -C /workspace/flash_attn_src
cd /workspace/flash_attn_src/flash_attn_npu-0.2.0b1
FLASH_ATTN_BUILD_VERSION=v3 FLASH_ATTENTION_FORCE_BUILD=TRUE \
  python -m pip install -v --no-build-isolation --no-deps .
```

也可以不安装 Python 包、直接使用源码目录，但编译扩展仍必须存在。不要把
`/absolute/path/to/...` 之类的示例占位符原样复制；先进入真实仓库，再使用 `pwd`：

```bash
cd /服务器上的真实路径/flash-attention-npu
export FLASH_ATTN_NPU_REPO="$(pwd)"
export PYTHONPATH="$FLASH_ATTN_NPU_REPO:$PYTHONPATH"
test -f "$FLASH_ATTN_NPU_REPO/flash_attn_npu_v3/flash_attn_interface.py"
```

用随附诊断脚本验证 ATK 当前环境：

```bash
python check_flash_attn_npu_v3_import.py
```

## 2. 生成 ATK 用例

在本目录运行：

```bash
atk case -f op_flash_attn_func_v3.yaml
```

生成文件通常位于：

```text
result/op_flash_attn_func_v3/json/all_op_flash_attn_func_v3.json
```

## 3. 执行精度测试

```bash
atk node --backend npu --devices 0 \
    node --backend cpu \
    task \
    -c result/op_flash_attn_func_v3/json/all_op_flash_attn_func_v3.json \
    --task accuracy \
    -p function_flash_attn_func_v3.py \
    --print_data output
```

如果当前 ATK 生成的目录名与上面不同，以 `atk case` 的控制台输出为准。

## 说明

- YAML 使用自定义 `api_type: flash_attn_npu_v3`；注册名必须与 Python 插件中的
  `@register("flash_attn_npu_v3")` 完全一致。
- 当前用例只验证前向、fp16、causal 路径。要验证 bf16，可把三个输入的
  `dtypes.values` 同时改成 `[bf16]`。
- 若要测非 causal，把 `causal.ranges.valid.values` 改为 `[false]`。
- 插件的 CPU 参考实现还覆盖 bottom-right causal mask、滑窗、softcap 和 MQA/GQA；
  `qv`、FP8 descale 与 `attention_chunk` 没有作为当前 CPU 标杆实现。
- `flash_attn_npu_v3` 在该项目中面向 910B/C；不要在只导出 KV-cache 接口的 950 路径上运行此用例。

## `ModuleNotFoundError` 对照

| 报错中的模块 | 含义 | 处理 |
|---|---|---|
| `flash_attn_npu_v3` | Python 包对 ATK worker 不可见 | 在同一 Python 环境安装，或设置 `FLASH_ATTN_NPU_REPO/PYTHONPATH` |
| `flash_attn_npu_3` | 找到了 Python 包，但 v3 编译扩展缺失 | 使用 `FLASH_ATTN_BUILD_VERSION=v3` 重新编译安装 |
| `torch_npu` | ATK worker 环境不是 NPU PyTorch 环境 | 激活正确环境并安装匹配的 torch_npu |
| `atk` 相关模块 | 插件被另一个 Python 直接运行 | 使用安装了 ATK 的解释器执行 |
