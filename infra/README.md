# Infra Etapu 22 — workspace Azure Databricks

Lokalny stan (`infra/terraform.tfstate`, poza gitem). Plan i apply uruchamia Kamil.
Stan underwritera (`etap13.tfstate`) nie ma tu żadnego zastosowania — nie mieszać.

## Jednorazowo (PowerShell)

```powershell
$AZ = "C:\Program Files\Microsoft SDKs\Azure\CLI2\wbin\az.cmd"
& $AZ provider register --namespace Microsoft.Databricks --wait
winget install Databricks.DatabricksCLI
```

## Utworzenie workspace'u

W PowerShellu argument `-out=...` z kropką musi być w apostrofach, inaczej PowerShell go rozbija ("Too many command line arguments", 31.10).

```powershell
$TF = "C:\Users\katar\Desktop\kopia_usb_rally\projekty\databricks_szkody_demo\infra"
terraform -chdir="$TF" init
terraform -chdir="$TF" plan '-out=etap22.tfplan'   # oczekiwane: 2 to add, 0 change, 0 destroy
terraform -chdir="$TF" apply etap22.tfplan
```

Sku `premium` (domyślne). `trial` odrzucony 31.10 (`TrialSkuNotSupportedForSubscriptionType` — Free Trial
go nie obsługuje), więc DBU idą z kredytu — sprawdzić Cost Management po pierwszym biegu joba.

## Usunięcie (08.11, razem z resztą Azure)

```powershell
terraform -chdir="$TF" destroy   # oczekiwane: 0 to add, 0 change, 2 to destroy
& $AZ group exists -n rg-databricks-demo-managed   # ma zwrócić false
```

Managed RG (`rg-databricks-demo-managed`) znika razem z workspace'em; jeśli zostanie — usunąć ręcznie i sprawdzić Cost Management.
