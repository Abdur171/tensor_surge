output "kubernetes_cluster_name" {
  value       = azurerm_kubernetes_cluster.aks.name
  description = "Name of the provisioned AKS cluster"
}

output "resource_group_name" {
  value       = azurerm_resource_group.rg.name
  description = "Resource group holding cluster resources"
}

output "gpu_node_pool_id" {
  value       = azurerm_kubernetes_cluster_node_pool.gpu_pool.id
  description = "Identifier of GPU node pool"
}
