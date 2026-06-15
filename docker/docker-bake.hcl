variable "REGISTRY" {
  default = "deeplabcut/deeplabcut"
}
variable "DLC_VERSION" {
  default = "3.0.0rc14"
}
variable "CUDA_VERSION" {
  default = "12.4"
}
variable "CUDNN_VERSION" {
  default = "9"
}
variable "PYTORCH_VERSION" {
  default = "2.5.1"
}
variable "MARK_LATEST" {
  default = false
}
target "_common" {
  context    = "."
  dockerfile = "Dockerfile"
  args = {
    DEEPLABCUT_VERSION = DLC_VERSION
    CUDA_VERSION       = CUDA_VERSION
    CUDNN_VERSION      = CUDNN_VERSION
    PYTORCH_VERSION    = PYTORCH_VERSION
  }
}
target "core" {
  inherits = ["_common"]
  target   = "core"
  tags = concat(
    ["${REGISTRY}:${DLC_VERSION}-core-cuda${CUDA_VERSION}"],
    MARK_LATEST ? ["${REGISTRY}:latest"] : []
  )
}
target "jupyter" {
  inherits = ["_common"]
  target   = "jupyter"
  tags = concat(
    ["${REGISTRY}:${DLC_VERSION}-jupyter-cuda${CUDA_VERSION}"],
    MARK_LATEST ? ["${REGISTRY}:latest-jupyter"] : []
  )
}
group "default" {
  targets = ["core", "jupyter"]
}
