variable "prefix" {
  type        = string
  default     = "tensor-surge"
  description = "Resource prefix identifier"
}

variable "location" {
  type        = string
  default     = "swedencentral"
  description = "Azure region for deployment"
}

variable "gpu_vm_size" {
  type        = string
  default     = "Standard_NC4as_T4_v3"
  description = "GPU Virtual Machine SKU for inference nodes"
}

variable "gpu_node_count" {
  type        = number
  default     = 2
  description = "Initial number of GPU nodes"
}
