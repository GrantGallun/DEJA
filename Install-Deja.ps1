param([switch]$NoLaunch)
# Adds deja (the replay player settings app) to the Start menu and opens it.
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
$storeOnly = $false
foreach ($dejaCandidate in ($dejaCandidates | Select-Object -Unique)) {
    $exe = & $dejaCandidate -c 'import sys, webview, smartreplay.frontend.window; print(sys.executable)' 2>$null
    if ($LASTEXITCODE -ne 0 -or !$exe) { continue }
    if ($exe -like '*\WindowsApps\*') { $storeOnly = $true; continue }
    $candidateW = Join-Path (Split-Path $exe) 'pythonw.exe'
    if (Test-Path -LiteralPath $candidateW) { $pythonw = $candidateW; break }
}
if (!$pythonw -and $storeOnly) { throw 'Only the Microsoft Store Python has pywebview, and its windows show the Python icon. Install Python from python.org, then: python -m pip install pywebview' }
if (!$pythonw) { throw 'Install pywebview first: python -m pip install pywebview. Or set SMARTREPLAY_PYTHON to a Python that has it.' }

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

[Deja.Shortcut]::Save($shortcut, $pythonw, '-m smartreplay.frontend --window', $PSScriptRoot, $icon, 'deja: replay player settings', $appId)
Write-Host "Added deja to the Start menu: $shortcut"
Write-Host 'To keep it on the taskbar, right-click its taskbar button and choose Pin to taskbar.'
if (!$NoLaunch) { Start-Process -FilePath $shortcut }
