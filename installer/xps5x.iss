; ============================================================================
;  Raeen - Windows installer (Inno Setup 6)
;
;  A sleek, per-user (non-elevated) setup wizard:
;    * Welcome / License (GPL-2.0) / install location
;    * a custom "choose your games folder" page
;    * optional desktop icon, windowed first-run, launch on finish
;    * bundles + silently installs the VC++ runtime when it's missing
;    * writes a ready-to-use config.toml pointing at the chosen games folder
;
;  Build it with installer\build.ps1 (which passes /DMyAppVersion and stages
;  the payload), or directly:  ISCC.exe /DMyAppVersion=0.1.0 installer\xps5x.iss
;
;  WHY PER-USER: the app reads and writes everything (config.toml, logs\,
;  savedata\, shader_cache\, themes\, the games scan) relative to its working
;  directory. A per-user install into a writable location - with every shortcut
;  pinned to WorkingDir={app} - keeps all of that working without elevation.
;  An all-users / Program Files install would need a small app change to
;  redirect user-data to %LOCALAPPDATA%; see installer\README.md.
; ============================================================================

#define MyAppName "Raeen"
#define MyAppPublisher "Raeen Project"
#define MyAppURL "https://github.com/Whoisraeen/Raeen"
#define MyAppExeName "raeen.exe"

; Version is normally injected by build.ps1 (from the workspace Cargo.toml).
; Falls back to this when compiling the .iss by hand.
#ifndef MyAppVersion
  #define MyAppVersion "0.1.0"
#endif

; Paths are resolved relative to this script's directory (installer\).
#define RepoRoot ".."
#define PayloadDir RepoRoot + "\target\release"

; Optional: drop the Microsoft VC++ 2015-2022 x64 redist at
; installer\redist\vc_redist.x64.exe and it is bundled + run when needed.
#define RedistSrc AddBackslash(SourcePath) + "redist\vc_redist.x64.exe"
#if FileExists(RedistSrc)
  #define HaveRedist
#endif

[Setup]
; A stable, unique AppId - never change it across releases (ties upgrades
; and the uninstaller together).
AppId={{B1E4C0A2-5D3F-4E7A-9C21-7F8A2D6E4B10}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}/issues
AppUpdatesURL={#MyAppURL}/releases
VersionInfoVersion={#MyAppVersion}
VersionInfoCompany={#MyAppPublisher}
VersionInfoProductName={#MyAppName}

; --- per-user, non-elevated -------------------------------------------------
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=auto
AllowNoIcons=yes
UsePreviousAppDir=yes

; --- platform ---------------------------------------------------------------
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

; --- pages / branding -------------------------------------------------------
WizardStyle=modern
DisableWelcomePage=no
LicenseFile={#RepoRoot}\LICENSE
WizardImageFile=assets\wizard-large.bmp
WizardSmallImageFile=assets\wizard-small.bmp
SetupIconFile=assets\xps5x.ico
UninstallDisplayIcon={app}\xps5x.ico
UninstallDisplayName={#MyAppName}

; --- output / packaging -----------------------------------------------------
OutputDir={#RepoRoot}\dist
OutputBaseFilename=Raeen-{#MyAppVersion}-Setup
Compression=lzma2/ultra64
SolidCompression=yes
SetupMutex=Raeen_Setup_Mutex

; --- behaviour --------------------------------------------------------------
; Offer to close a running Raeen so its files aren't locked mid-install.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"
Name: "windowedmode"; Description: "Start in a resizable window instead of full-screen"; GroupDescription: "First-run options:"; Flags: unchecked

[Dirs]
; Runtime working directories, created up-front so the first launch is clean.
; savedata is user data - never wiped on uninstall (see [UninstallDelete]).
Name: "{app}\logs"
Name: "{app}\savedata"
Name: "{app}\shader_cache"
Name: "{app}\firmware"

[Files]
Source: "{#PayloadDir}\{#MyAppExeName}"; DestDir: "{app}"; Flags: ignoreversion
; Branded icon for shortcuts + Add/Remove Programs (the exe has no embedded
; icon yet - see installer\README.md "Follow-ups").
Source: "assets\xps5x.ico"; DestDir: "{app}"; Flags: ignoreversion
; The on-disk default theme (the Shell resolves themes\<name>\theme.toml
; relative to its working directory).
Source: "{#RepoRoot}\themes\default\theme.toml"; DestDir: "{app}\themes\default"; Flags: ignoreversion
; Docs.
Source: "{#RepoRoot}\LICENSE"; DestDir: "{app}"; DestName: "LICENSE.txt"; Flags: ignoreversion
Source: "{#RepoRoot}\THIRD_PARTY_NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion
Source: "{#RepoRoot}\README.md"; DestDir: "{app}"; DestName: "README.md"; Flags: ignoreversion
#ifdef HaveRedist
Source: "{#RedistSrc}"; DestDir: "{tmp}"; Flags: deleteafterinstall
#endif

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\xps5x.ico"; Comment: "Launch Raeen"
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; WorkingDir: "{app}"; IconFilename: "{app}\xps5x.ico"; Comment: "Launch Raeen"; Tasks: desktopicon

[Run]
#ifdef HaveRedist
Filename: "{tmp}\vc_redist.x64.exe"; Parameters: "/install /quiet /norestart"; StatusMsg: "Installing the Visual C++ runtime..."; Check: VCRedistNeeded; Flags: waituntilterminated
#endif
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; WorkingDir: "{app}"; Flags: nowait postinstall skipifsilent
Filename: "{code:GetGamesDir}"; Description: "Open my games folder"; Flags: postinstall shellexec skipifsilent unchecked

[UninstallDelete]
; Only regenerable data is removed. User saves and config are left in place.
Type: filesandordirs; Name: "{app}\shader_cache"
Type: filesandordirs; Name: "{app}\logs"

; ============================================================================
[Code]
var
  GamesDirPage: TInputDirWizardPage;

{ True unless the VC++ 2015-2022 x64 runtime is already registered as
  installed. Checks both registry views; HKLM reads need no elevation. }
function VCRedistNeeded(): Boolean;
var
  Installed: Cardinal;
begin
  Result := True;
  if RegQueryDWordValue(HKLM64, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64', 'Installed', Installed) and (Installed = 1) then
    Result := False
  else if RegQueryDWordValue(HKLM32, 'SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\X64', 'Installed', Installed) and (Installed = 1) then
    Result := False;
end;

{ The games folder the user picked - used by the "Open my games folder" [Run]
  entry and the config.toml written at post-install. }
function GetGamesDir(Param: String): String;
begin
  Result := Trim(GamesDirPage.Values[0]);
end;

procedure InitializeWizard();
begin
  GamesDirPage := CreateInputDirPage(wpSelectDir,
    'Select your games folder',
    'Where should Raeen look for your games?',
    'Raeen scans this folder for PS5 titles. Put each game in its own '  +
    'sub-folder containing an eboot.bin. You can add or change folders '  +
    'any time later in Settings > Game Folders.' + #13#10 + #13#10 +
    'Choose a folder, then click Next.',
    False, 'Games');
  GamesDirPage.Add('');
  GamesDirPage.Values[0] := ExpandConstant('{userdocs}\Raeen\Games');
end;

function NextButtonClick(CurPageID: Integer): Boolean;
var
  Dir: String;
begin
  Result := True;
  if CurPageID = GamesDirPage.ID then
  begin
    Dir := Trim(GamesDirPage.Values[0]);
    if Dir = '' then
    begin
      MsgBox('Please choose a games folder.', mbError, MB_OK);
      Result := False;
      Exit;
    end;
    { Create it now so the choice is real and we can confirm it is writable. }
    if not DirExists(Dir) then
    begin
      if not ForceDirectories(Dir) then
      begin
        MsgBox('Raeen could not create this folder:' + #13#10 + Dir + #13#10 + #13#10 +
               'Please choose a different location.', mbError, MB_OK);
        Result := False;
        Exit;
      end;
    end;
  end;
end;

{ Add the chosen games folder to the "Ready to Install" summary. }
function UpdateReadyMemo(Space, NewLine, MemoUserInfoInfo, MemoDirInfo, MemoTypeInfo,
  MemoComponentsInfo, MemoGroupInfo, MemoTasksInfo: String): String;
begin
  Result := MemoDirInfo + NewLine + NewLine +
            'Games folder:' + NewLine + Space + Trim(GamesDirPage.Values[0]) + NewLine;
  if MemoGroupInfo <> '' then
    Result := Result + NewLine + MemoGroupInfo + NewLine;
  if MemoTasksInfo <> '' then
    Result := Result + NewLine + MemoTasksInfo;
end;

{ Write a complete, hand-editable config.toml into the install dir, pointing the
  library scan at the chosen games folder. Only on a fresh install - an existing
  config.toml (upgrade / reinstall) is left untouched so user settings survive. }
procedure WriteConfig();
var
  ConfigPath, Games, Fullscreen, Toml: String;
begin
  ConfigPath := ExpandConstant('{app}\config.toml');
  if FileExists(ConfigPath) then
    Exit;

  Games := Trim(GamesDirPage.Values[0]);
  { TOML literal strings (single quotes) take backslashes verbatim, so Windows
    paths need no escaping - but a literal single quote would end the string, so
    strip any (paths practically never contain one). }
  StringChangeEx(Games, '''', '', True);

  if WizardIsTaskSelected('windowedmode') then
    Fullscreen := 'false'
  else
    Fullscreen := 'true';

  Toml :=
    '# Raeen configuration - generated by the installer. Safe to edit.' + #13#10 +
    '# Paths are relative to this file (the install directory).' + #13#10 + #13#10 +
    '[general]' + #13#10 +
    'fullscreen = ' + Fullscreen + #13#10 +
    'window_width = 1920' + #13#10 +
    'window_height = 1080' + #13#10 +
    'vsync = true' + #13#10 +
    'selected_theme = "default"' + #13#10 + #13#10 +
    '[graphics]' + #13#10 +
    'backend = "Vulkan"' + #13#10 +
    'resolution_scale = 1.0' + #13#10 +
    'shader_cache = true' + #13#10 +
    'gpu_device_index = 0' + #13#10 +
    'validation_layers = false' + #13#10 + #13#10 +
    '[audio]' + #13#10 +
    'enabled = true' + #13#10 +
    'volume = 1.0' + #13#10 +
    'spatial_audio = true' + #13#10 + #13#10 +
    '[input]' + #13#10 +
    'dualsense_features = true' + #13#10 +
    'deadzone = 0.15' + #13#10 + #13#10 +
    '[debug]' + #13#10 +
    'logging = true' + #13#10 +
    'log_level = "info"' + #13#10 +
    'dump_gpu_commands = false' + #13#10 +
    'dump_shaders = false' + #13#10 +
    'trace_syscalls = false' + #13#10 + #13#10 +
    '[paths]' + #13#10 +
    'games_dir = "games"' + #13#10 +
    'firmware_dir = "firmware"' + #13#10 +
    'save_dir = "savedata"' + #13#10 +
    'shader_cache_dir = "shader_cache"' + #13#10 +
    'log_dir = "logs"' + #13#10 +
    'game_folders = [''' + Games + ''']' + #13#10 +
    'key_provider_path = ""' + #13#10;

  if not SaveStringToFile(ConfigPath, Toml, False) then
    MsgBox('Raeen could not write its configuration file:' + #13#10 + ConfigPath + #13#10 + #13#10 +
           'The app will create a default one on first launch; set your games ' +
           'folder in Settings > Game Folders.', mbInformation, MB_OK);
end;

procedure CurStepChanged(CurStep: TSetupStep);
begin
  if CurStep = ssPostInstall then
    WriteConfig();
end;
