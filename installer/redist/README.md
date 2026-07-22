# Bundled redistributables

Drop **`vc_redist.x64.exe`** here to bundle the Microsoft Visual C++ 2015-2022
x64 runtime into the installer.

```powershell
# Download it straight into this folder:
Invoke-WebRequest https://aka.ms/vs/17/release/vc_redist.x64.exe -OutFile installer\redist\vc_redist.x64.exe
```

When present, `raeen.iss` bundles it and runs it silently **only if** the
runtime is missing on the target machine. When absent, the step is compiled out
and setup still works (assuming the runtime is already installed).

> The redist binary is Microsoft's and is intentionally **not** committed to the
> repo — fetch it at package time.
