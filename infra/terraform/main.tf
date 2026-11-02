terraform {
  required_version = ">= 1.5.0"
  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.90.0"
    }
  }
}

provider "azurerm" {
  features {}
}

resource "azurerm_resource_group" "rg" {
  name     = "${var.prefix}-rg"
  location = var.location
}

# Azure Kubernetes Service (AKS) Cluster with GPU Support
resource "azurerm_kubernetes_cluster" "aks" {
  name                = "${var.prefix}-aks"
  location            = azurerm_resource_group.rg.location
  resource_group_name = azurerm_resource_group.rg.name
  dns_prefix          = "${var.prefix}-k8s"

  default_node_pool {
    name       = "systempool"
    node_count = 2
    vm_size    = "Standard_D4s_v5"
    os_disk_size_gb = 64
  }

  identity {
    type = "SystemAssigned"
  }

  network_profile {
    network_plugin    = "azure"
    load_balancer_sku = "standard"
  }
}

# Dedicated GPU Node Pool for Triton & Model Serving
resource "azurerm_kubernetes_cluster_node_pool" "gpu_pool" {
  name                  = "gpunodes"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.aks.id
  vm_size               = var.gpu_vm_size
  node_count            = var.gpu_node_count
  min_count             = 1
  max_count             = 8
  enable_auto_scaling   = true

  node_labels = {
    "accelerator" = "nvidia"
    "workload"    = "inference"
  }

  node_taints = [
    "sku=gpu:NoSchedule"
  ]
}
