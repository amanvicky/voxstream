; =====================================================================
;  VoxStream v2.0.0 - Inno Setup installer script
;  Builds: release\VoxStream-Setup-v2.0.exe
;
;  Installs the standalone dist\VoxStream bundle per-user (no admin
;  required): with PrivilegesRequired=lowest, {autopf} automatically
;  resolves to {localappdata}\Programs\VoxStream.
;
;  Compile: ISCC.exe installer\VoxStream.iss
; =====================================================================

#define MyAppName "VoxStream"
#define MyAppVersion "2.0.0"
#define MyAppPublisher "Elugam Aman"
#define MyAppURL "https://github.com/amanvicky/voxstream"
#define MyAppExeName "VoxStream.exe"

[Setup]
AppId={{8E4F2A31-9C7B-4E52-B4D8-2F6A5C7D9013}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir=..\release
OutputBaseFilename=VoxStream-Setup-v2.0
SetupIconFile=app_icon.ico
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayIcon={app}\{#MyAppExeName}
ChangesEnvironment=no
MinVersion=10.0

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a &Desktop Shortcut"; \
    GroupDescription: "Shortcuts:"; Flags: checkedonce
Name: "autostart"; Description: "Start VoxStream automatically with Windows"; \
    GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\VoxStream\*"; DestDir: "{app}"; \
    Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Comment: "VoxStream - offline voice typing"
Name: "{group}\Uninstall {#MyAppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; \
    Tasks: desktopicon; Comment: "VoxStream - offline voice typing"

[Registry]
; Start-with-Windows task (per user; removed again on uninstall).
Root: HKCU; Subkey: "Software\Microsoft\Windows\CurrentVersion\Run"; \
    ValueType: string; ValueName: "VoxStream"; \
    ValueData: """{app}\{#MyAppExeName}"""; \
    Tasks: autostart; Flags: uninsdeletevalue

[Run]
Filename: "{app}\{#MyAppExeName}"; \
    Description: "{cm:LaunchProgram,{#MyAppName}}"; \
    Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Writable runtime data created next to the app (logs, history, ui state).
Type: filesandordirs; Name: "{app}\logs"
Type: filesandordirs; Name: "{app}\data"
