// Etap 22 - workspace Azure Databricks dla mini-demo (SPEC_ETAP22.md, decyzja 2).
//
// Osobny, LOKALNY tfstate tego repo (terraform.tfstate w infra/, poza gitem).
// Nie ma nic wspolnego z etap13.tfstate underwritera - tamten stan jest wspolny
// dla kilku etapow i pelny apply/destroy na nim niszczy zasoby.
terraform {
  required_version = ">= 1.9"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 5.10"
    }
  }
}

provider "azurerm" {
  features {}
}

variable "lokalizacja" {
  type    = string
  default = "swedencentral"
}

// "premium" (Unity Catalog wymaga Premium; "standard" odpada). "trial" odrzucony 31.10:
// TrialSkuNotSupportedForSubscriptionType - niedostepny na subskrypcji Azure Free Trial.
// DBU placone z kredytu trialu.
variable "sku" {
  type    = string
  default = "premium"
}

locals {
  tagi = {
    projekt = "databricks_szkody_demo"
    etap    = "22"
    usunac  = "2026-11-08"
  }
}

resource "azurerm_resource_group" "demo" {
  name     = "rg-databricks-demo"
  location = var.lokalizacja
  tags     = local.tagi
}

resource "azurerm_databricks_workspace" "demo" {
  name                        = "dbw-szkody-demo-kk1"
  resource_group_name         = azurerm_resource_group.demo.name
  location                    = azurerm_resource_group.demo.location
  sku                         = var.sku
  managed_resource_group_name = "rg-databricks-demo-managed"
  tags                        = local.tagi
}

output "url_workspace" {
  value = "https://${azurerm_databricks_workspace.demo.workspace_url}"
}

output "managed_rg" {
  value = azurerm_databricks_workspace.demo.managed_resource_group_name
}
