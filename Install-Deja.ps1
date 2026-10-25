param([switch]$NoLaunch)
# Adds deja (settings, automatic capture and replay) to the Start menu and opens it.
# The shortcut carries deja's icon and the AppUserModelID the window claims
# (smartreplay/frontend/window.py), so its taskbar button shows the logo and
# "Pin to taskbar" pins this shortcut rather than Python.
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$appId = 'SmartReplay.Deja'
$icon = Join-Path $PSScriptRoot 'smartreplay\frontend\static\deja.ico'
$shortcut = Join-Path ([Environment]::GetFolderPath('Programs')) 'deja.lnk'

# A Python that has pywebview and can import this checkout. Not the Microsoft
# Store build: it runs as a packaged app, and Windows files a packaged app's
# windows under the package's identity and icon, ignoring the ID deja claims.
$dejaCandidates = @()
if ($env:SMARTREPLAY_PYTHON) { $dejaCandidates += $env:SMARTREPLAY_PYTHON }
$dejaCandidates += @(Get-ChildItem -Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe", "$env:ProgramFiles\Python3*\python.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending | Select-Object -ExpandProperty FullName)
$dejaCommand = Get-Command python -ErrorAction SilentlyContinue
if ($dejaCommand) { $dejaCandidates += $dejaCommand.Source }
$pythonw = $null
$dejaInstallPython = $null
$storeOnly = $false
foreach ($dejaCandidate in ($dejaCandidates | Select-Object -Unique)) {
    try { $exe = & $dejaCandidate -c 'import sys; print(sys.executable if sys.version_info >= (3, 11) else "")' 2>$null } catch { continue }
    if ($LASTEXITCODE -ne 0 -or !$exe) { continue }
    if ($exe -like '*\WindowsApps\*') { $storeOnly = $true; continue }
    $candidateW = Join-Path (Split-Path $exe) 'pythonw.exe'
    if (!(Test-Path -LiteralPath $candidateW)) { continue }
    if (!$dejaInstallPython) { $dejaInstallPython = $exe }
    try {
        & $exe -c 'import webview, cv2, scipy, PIL, imageio_ffmpeg, faster_whisper, anthropic, smartreplay.demo.server, smartreplay.frontend.window' 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonw = $candidateW; break }
    } catch { }
}
if (!$pythonw -and $dejaInstallPython) {
    Write-Host 'Installing deja recording, voice and window dependencies...'
    & $dejaInstallPython -m pip install -r (Join-Path $PSScriptRoot 'requirements-demo.txt') pywebview
    if ($LASTEXITCODE -ne 0) { throw 'Deja dependencies could not be installed. Check the installation output above.' }
    & $dejaInstallPython -c 'import webview, cv2, scipy, PIL, imageio_ffmpeg, faster_whisper, anthropic, smartreplay.demo.server, smartreplay.frontend.window'
    if ($LASTEXITCODE -ne 0) { throw 'Deja dependencies were installed but could not load.' }
    $pythonw = Join-Path (Split-Path $dejaInstallPython) 'pythonw.exe'
}
if (!$pythonw -and $storeOnly) { throw 'Install Python 3.11 or later from python.org. The Microsoft Store build does not support deja taskbar identity.' }
if (!$pythonw) { throw 'Install Python 3.11 or later from python.org, or set SMARTREPLAY_PYTHON to its python.exe.' }
$dejaSpeechPython = Join-Path (Split-Path $pythonw) 'python.exe'
Write-Host 'Preparing local microphone recognition...'
& $dejaSpeechPython -c 'import os; from faster_whisper.utils import download_model; download_model(os.environ.get("SMARTREPLAY_VOICE_CPU_MODEL", "base.en"))'
if ($LASTEXITCODE -ne 0) { throw 'The local microphone model could not be installed. Check your connection and run the installer again.' }
$dejaNative = if ($env:SMARTREPLAY_NATIVE_EXE) { $env:SMARTREPLAY_NATIVE_EXE } else { Join-Path $PSScriptRoot 'experiments\gamehq-passive\out-passive\GameHQPassiveReplay.exe' }
if (!(Test-Path -LiteralPath $dejaNative -PathType Leaf)) {
    if ($env:SMARTREPLAY_NATIVE_EXE) { throw "The configured native player does not exist: $dejaNative" }
    Write-Host 'Building the deja native player...'
    # Run in a separate PowerShell: the native build script exits its shell.
    & powershell.exe -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot 'experiments\gamehq-passive\spike\passive-replay\build.ps1')
    if ($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $dejaNative -PathType Leaf)) { throw 'The native player could not be built. Check that the GameHQ checkout and its Qt/CMake/MinGW toolchain are installed.' }
}

Add-Type -TypeDefinition @'
using System;
using System.Runtime.InteropServices;
using System.Runtime.InteropServices.ComTypes;
using System.Text;

namespace Deja {
    [ComImport, Guid("000214F9-0000-0000-C000-000000000046"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IShellLinkW {
        void GetPath([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder file, int size, IntPtr data, uint flags);
        void GetIDList(out IntPtr idList);
        void SetIDList(IntPtr idList);
        void GetDescription([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder name, int size);
        void SetDescription([MarshalAs(UnmanagedType.LPWStr)] string name);
        void GetWorkingDirectory([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder dir, int size);
        void SetWorkingDirectory([MarshalAs(UnmanagedType.LPWStr)] string dir);
        void GetArguments([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder args, int size);
        void SetArguments([MarshalAs(UnmanagedType.LPWStr)] string args);
        void GetHotkey(out short hotkey);
        void SetHotkey(short hotkey);
        void GetShowCmd(out int showCmd);
        void SetShowCmd(int showCmd);
        void GetIconLocation([Out, MarshalAs(UnmanagedType.LPWStr)] StringBuilder path, int size, out int index);
        void SetIconLocation([MarshalAs(UnmanagedType.LPWStr)] string path, int index);
        void SetRelativePath([MarshalAs(UnmanagedType.LPWStr)] string path, uint reserved);
        void Resolve(IntPtr hwnd, uint flags);
        void SetPath([MarshalAs(UnmanagedType.LPWStr)] string file);
    }

    [StructLayout(LayoutKind.Sequential, Pack = 4)]
    public struct PropertyKey { public Guid FormatId; public uint PropertyId; }

    // PROPVARIANT holding a string: the type tag, then the pointer at offset 8.
    [StructLayout(LayoutKind.Explicit)]
    public struct PropVariant {
        [FieldOffset(0)] public ushort Type;
        [FieldOffset(8)] public IntPtr Pointer;
    }

    [ComImport, Guid("886D8EEB-8CF2-4446-8D02-CDBA1DBDCF99"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
    interface IPropertyStore {
        void GetCount(out uint count);
        void GetAt(uint index, out PropertyKey key);
        void GetValue(ref PropertyKey key, out PropVariant value);
        void SetValue(ref PropertyKey key, ref PropVariant value);
        void Commit();
    }

    [ComImport, Guid("00021401-0000-0000-C000-000000000046")]
    class ShellLink { }

    public static class Shortcut {
        // System.AppUserModel.ID
        static PropertyKey AppIdKey = new PropertyKey { FormatId = new Guid("9F4C2855-9F79-4B39-A8D0-E1D42DE1D5F3"), PropertyId = 5 };

        public static void Save(string path, string target, string args, string dir, string icon, string description, string appId) {
            var link = (IShellLinkW)new ShellLink();
            link.SetPath(target);
            link.SetArguments(args);
            link.SetWorkingDirectory(dir);
            link.SetIconLocation(icon, 0);
            link.SetDescription(description);
            var value = new PropVariant { Type = 31 /* VT_LPWSTR */, Pointer = Marshal.StringToCoTaskMemUni(appId) };
            try {
                var store = (IPropertyStore)link;
                store.SetValue(ref AppIdKey, ref value);
                store.Commit();
            } finally {
                Marshal.FreeCoTaskMem(value.Pointer);
            }
            ((IPersistFile)link).Save(path, true);
        }
    }
}
'@

[Deja.Shortcut]::Save($shortcut, $pythonw, '-m smartreplay.frontend --window', $PSScriptRoot, $icon, 'deja: automatic game recording and replay', $appId)
Write-Host "Added deja to the Start menu: $shortcut"
Write-Host 'To keep it on the taskbar, right-click its taskbar button and choose Pin to taskbar.'
if (!$NoLaunch) { Start-Process -FilePath $shortcut }
