Dim WshShell, oShortcut, strDesktop, strProjectDir

Set WshShell = CreateObject("WScript.Shell")
strDesktop = WshShell.SpecialFolders("Desktop")
strProjectDir = CreateObject("Scripting.FileSystemObject").GetParentFolderName(WScript.ScriptFullName)

Set oShortcut = WshShell.CreateShortcut(strDesktop & "\AI Stock Agent.lnk")
oShortcut.TargetPath = "wscript.exe"
oShortcut.Arguments = """" & strProjectDir & "\silent_run.vbs"""
oShortcut.WorkingDirectory = strProjectDir
oShortcut.WindowStyle = 7
oShortcut.Description = "LEVI FINANCE - AI Stock Agent"
oShortcut.Save

Set oShortcut = Nothing
Set WshShell = Nothing

MsgBox "הקיצור 'AI Stock Agent' נוצר בהצלחה בשולחן העבודה!", 64, "LEVI FINANCE"
