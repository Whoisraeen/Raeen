# Generates the exact AGC version-10 compact register-default data used for
# guest API versions 10 and 12. Source: KytyPS5 (MIT, InoriRus/Nmzik).
param(
    [string]$Source = "reference/kytyps5/src/libs/agcRegisterDefaults.inc",
    [string]$Output = "crates/raeen-hle/src/libsce_agc_reg_defaults_v10.rs"
)

$text = Get-Content -LiteralPath $Source -Raw

function Get-RegisterArray([string]$name) {
    $pattern = "static ShaderRegister\s+$([regex]::Escape($name))\[\]\s*=\s*\{(?<body>.*?)\};"
    $match = [regex]::Match($text, $pattern, [Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $match.Success) { return @() }
    return [regex]::Matches($match.Groups["body"].Value, "\{\s*(0x[0-9a-fA-F]+)\s*,\s*(0x[0-9a-fA-F]+)\s*\}") |
        ForEach-Object { "    ($($_.Groups[1].Value), $($_.Groups[2].Value))," }
}

function Get-ScalarArray([string]$name, [string]$type) {
    $pattern = "static(?: const)?\s+$type\s+$([regex]::Escape($name))\[\]\s*=\s*\{(?<body>.*?)\};"
    $match = [regex]::Match($text, $pattern, [Text.RegularExpressions.RegexOptions]::Singleline)
    if (-not $match.Success) { return @() }
    return [regex]::Matches($match.Groups["body"].Value, "0x[0-9a-fA-F]+") |
        ForEach-Object { "    $($_.Value)," }
}

$lines = [Collections.Generic.List[string]]::new()
$lines.Add("//! Exact AGC v10 compact register defaults (also selected by API v12).")
$lines.Add("//!")
$lines.Add("//! Mechanically generated from KytyPS5 agcRegisterDefaults.inc")
$lines.Add("//! (MIT, (c) InoriRus/Nmzik). Do not hand-edit; run")
$lines.Add("//! tools/generate_agc_defaults_v10.ps1 after updating the reference.")
$lines.Add("")
$lines.Add("pub(crate) struct CompactRegisterDefaultsV10 {")
$lines.Add("    pub registers: [&'static [(u32, u32)]; 4],")
$lines.Add("    pub pointer_offsets: [&'static [u16]; 4],")
$lines.Add("    pub types: &'static [u32],")
$lines.Add("}")
$lines.Add("")

foreach ($scope in @("public", "internal")) {
    $upper = $scope.ToUpperInvariant()
    foreach ($table in 0..3) {
        $base = "g_agc_${scope}_reg_defaults_v10_tbl${table}"
        $regName = "${upper}_TBL${table}_REGS"
        $ptrName = "${upper}_TBL${table}_PTRS"
        $regs = @(Get-RegisterArray "${base}_regs")
        $ptrs = @(Get-ScalarArray "${base}_ptrs" "uint16_t")
        $lines.Add("#[rustfmt::skip]")
        $lines.Add("static ${regName}: &[(u32, u32)] = &[")
        foreach ($line in $regs) { $lines.Add($line) }
        $lines.Add("];")
        $lines.Add("#[rustfmt::skip]")
        $lines.Add("static ${ptrName}: &[u16] = &[")
        foreach ($line in $ptrs) { $lines.Add($line) }
        $lines.Add("];")
        $lines.Add("")
    }
    $typesName = "${upper}_TYPES"
    $types = @(Get-ScalarArray "g_agc_${scope}_reg_defaults_v10_types" "uint32_t")
    $lines.Add("#[rustfmt::skip]")
    $lines.Add("static ${typesName}: &[u32] = &[")
    foreach ($line in $types) { $lines.Add($line) }
    $lines.Add("];")
    $lines.Add("")
    $lines.Add("pub(crate) static ${upper}_V10: CompactRegisterDefaultsV10 = CompactRegisterDefaultsV10 {")
    $lines.Add("    registers: [${upper}_TBL0_REGS, ${upper}_TBL1_REGS, ${upper}_TBL2_REGS, ${upper}_TBL3_REGS],")
    $lines.Add("    pointer_offsets: [${upper}_TBL0_PTRS, ${upper}_TBL1_PTRS, ${upper}_TBL2_PTRS, ${upper}_TBL3_PTRS],")
    $lines.Add("    types: ${upper}_TYPES,")
    $lines.Add("};")
    $lines.Add("")
}

Set-Content -LiteralPath $Output -Value $lines -Encoding utf8
