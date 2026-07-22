<#
    gen-assets.ps1 - generate Raeen installer branding.

    Produces, under installer/assets/:
      * raeen.ico          multi-resolution app + setup icon (16..256, PNG frames)
      * wizard-large.bmp   Welcome/Finished page sidebar (portrait, full-bleed)
      * wizard-small.bmp   inner-page header logo (square-ish, white background)

    All art is drawn from scratch with System.Drawing (GDI+) so it is fully
    reproducible and carries no third-party assets - regenerate any time by
    running this script. Tweak the palette / glyph constants below to rebrand.

    Usage (Windows PowerShell 5.1+):
      powershell -ExecutionPolicy Bypass -File installer\scripts\gen-assets.ps1
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

# ---------------------------------------------------------------- paths -----
$AssetsDir = Join-Path (Split-Path -Parent $PSScriptRoot) 'assets'
if (-not (Test-Path $AssetsDir)) { New-Item -ItemType Directory -Path $AssetsDir | Out-Null }

# --------------------------------------------------------------- palette ----
# Deep-space navy -> electric blue badge, cyan accent. Swap these to rebrand.
$NavyDeep   = [System.Drawing.Color]::FromArgb(255, 12,  22,  58)   # #0C163A
$NavyMid    = [System.Drawing.Color]::FromArgb(255, 22,  44, 110)   # #162C6E
$Blue       = [System.Drawing.Color]::FromArgb(255, 42, 112, 255)   # #2A70FF
$Cyan       = [System.Drawing.Color]::FromArgb(255, 64, 224, 255)   # #40E0FF
$InkBottom  = [System.Drawing.Color]::FromArgb(255,  8,  12,  30)   # #080C1E
$InkTop     = [System.Drawing.Color]::FromArgb(255, 16,  30,  74)   # #101E4A
$White      = [System.Drawing.Color]::White
$GLYPH      = 'R'   # monogram drawn on the badge

# ------------------------------------------------------------- helpers ------
function New-RoundedPath([System.Drawing.RectangleF]$r, [single]$radius) {
    $d = $radius * 2.0
    $p = New-Object System.Drawing.Drawing2D.GraphicsPath
    $p.AddArc($r.X,             $r.Y,             $d, $d, 180, 90)
    $p.AddArc($r.Right - $d,    $r.Y,             $d, $d, 270, 90)
    $p.AddArc($r.Right - $d,    $r.Bottom - $d,   $d, $d,   0, 90)
    $p.AddArc($r.X,             $r.Bottom - $d,   $d, $d,  90, 90)
    $p.CloseFigure()
    return $p
}

# Draw the Raeen badge (rounded gradient square + monogram + gloss) filling a
# square region of side $S onto graphics $g at (0,0). Transparent outside the
# badge so it composites over any background.
function Draw-Badge([System.Drawing.Graphics]$g, [int]$S) {
    $g.SmoothingMode     = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.PixelOffsetMode   = [System.Drawing.Drawing2D.PixelOffsetMode]::HighQuality
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    $pad    = [single]([Math]::Max(1.0, $S * 0.055))
    $side   = [single]($S - 2 * $pad)
    $rect   = New-Object System.Drawing.RectangleF($pad, $pad, $side, $side)
    $radius = [single]($side * 0.235)
    $path   = New-RoundedPath $rect $radius

    # Diagonal navy -> blue fill.
    $grad = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $NavyDeep, $Blue, 55.0)
    $blend = New-Object System.Drawing.Drawing2D.ColorBlend(3)
    $blend.Colors    = @($NavyDeep, $NavyMid, $Blue)
    $blend.Positions = @(0.0, 0.55, 1.0)
    $grad.InterpolationColors = $blend
    $g.FillPath($grad, $path)

    # Top gloss: a soft white sheen fading down over the upper third.
    $glossRect = New-Object System.Drawing.RectangleF($pad, $pad, $side, [single]($side * 0.55))
    $gloss = New-Object System.Drawing.Drawing2D.LinearGradientBrush(
        $glossRect,
        [System.Drawing.Color]::FromArgb(70, 255, 255, 255),
        [System.Drawing.Color]::FromArgb(0, 255, 255, 255), 90.0)
    $clip = New-RoundedPath $rect $radius
    $g.SetClip($clip)
    $g.FillRectangle($gloss, $glossRect)
    $g.ResetClip()

    # Hairline inner highlight for a crisp edge.
    $penHi = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(90, 255, 255, 255), [single]([Math]::Max(1.0, $S * 0.010)))
    $g.DrawPath($penHi, $path)

    # Monogram, centered, in a heavy Segoe UI. Falls back gracefully if the
    # exact face is missing (GDI substitutes the family's bold).
    $fontSize = [single]($side * 0.44)
    $font = New-Object System.Drawing.Font('Segoe UI Black', $fontSize, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    if ($font.Name -ne 'Segoe UI Black') {
        $font.Dispose()
        $font = New-Object System.Drawing.Font('Segoe UI', $fontSize, [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    }
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment     = [System.Drawing.StringAlignment]::Center
    $fmt.LineAlignment = [System.Drawing.StringAlignment]::Center
    # Nudge up slightly so the cyan accent bar below reads as a base.
    $textRect = New-Object System.Drawing.RectangleF($pad, [single]($pad - $side * 0.02), $side, $side)
    # Subtle shadow, then a white-to-cyan fill for depth.
    $shadow = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(70, 0, 0, 0))
    $shRect = New-Object System.Drawing.RectangleF($textRect.X, [single]($textRect.Y + $side*0.012), $textRect.Width, $textRect.Height)
    $g.DrawString($GLYPH, $font, $shadow, $shRect, $fmt)
    $textBrush = New-Object System.Drawing.Drawing2D.LinearGradientBrush($rect, $White, $Cyan, 90.0)
    $g.DrawString($GLYPH, $font, $textBrush, $textRect, $fmt)

    # Cyan accent bar under the monogram - the "5" energy line.
    $barW = [single]($side * 0.34)
    $barH = [single]([Math]::Max(1.5, $side * 0.055))
    $barX = [single]($pad + ($side - $barW) / 2.0)
    $barY = [single]($pad + $side * 0.735)
    $barRect = New-Object System.Drawing.RectangleF($barX, $barY, $barW, $barH)
    $barPath = New-RoundedPath $barRect ([single]($barH / 2.0))
    $g.FillPath((New-Object System.Drawing.SolidBrush($Cyan)), $barPath)

    $grad.Dispose(); $gloss.Dispose(); $penHi.Dispose(); $font.Dispose()
    $shadow.Dispose(); $textBrush.Dispose(); $path.Dispose(); $clip.Dispose(); $barPath.Dispose()
}

function New-BadgeBitmap([int]$S) {
    $bmp = New-Object System.Drawing.Bitmap($S, $S, [System.Drawing.Imaging.PixelFormat]::Format32bppArgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear([System.Drawing.Color]::Transparent)
    Draw-Badge $g $S
    $g.Dispose()
    return $bmp
}

# --------------------------------------------------------------- icon -------
function Write-Ico([string]$path, [int[]]$sizes) {
    $frames = @()
    foreach ($s in $sizes) {
        $bmp = New-BadgeBitmap $s
        $ms = New-Object System.IO.MemoryStream
        $bmp.Save($ms, [System.Drawing.Imaging.ImageFormat]::Png)
        $frames += , ($ms.ToArray())
        $ms.Dispose(); $bmp.Dispose()
    }
    $fs = [System.IO.File]::Create($path)
    $bw = New-Object System.IO.BinaryWriter($fs)
    $bw.Write([UInt16]0); $bw.Write([UInt16]1); $bw.Write([UInt16]$sizes.Count)   # ICONDIR
    $offset = 6 + 16 * $sizes.Count
    for ($i = 0; $i -lt $sizes.Count; $i++) {
        $s = $sizes[$i]; $len = $frames[$i].Length
        $dim = if ($s -ge 256) { 0 } else { $s }
        $bw.Write([Byte]$dim); $bw.Write([Byte]$dim); $bw.Write([Byte]0); $bw.Write([Byte]0)
        $bw.Write([UInt16]1); $bw.Write([UInt16]32)
        $bw.Write([UInt32]$len); $bw.Write([UInt32]$offset)
        $offset += $len
    }
    foreach ($f in $frames) { $bw.Write($f) }
    $bw.Flush(); $fs.Close()
}

# ------------------------------------------------------- wizard sidebar -----
# Portrait, full-bleed dark art with the badge, wordmark, and tagline. 24bpp so
# Inno gets a plain BMP (no alpha channel to trip older readers).
function Write-WizardLarge([string]$path, [int]$W, [int]$H) {
    $bmp = New-Object System.Drawing.Bitmap($W, $H, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.SmoothingMode     = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $g.TextRenderingHint = [System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit

    # Vertical ink gradient.
    $full = New-Object System.Drawing.Rectangle(0, 0, $W, $H)
    $bg = New-Object System.Drawing.Drawing2D.LinearGradientBrush($full, $InkTop, $InkBottom, 90.0)
    $g.FillRectangle($bg, $full)

    # Faint diagonal grid lines for texture.
    $penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(14, 120, 170, 255), 1.0)
    for ($x = - $H; $x -lt $W; $x += [int]($W * 0.11)) {
        $g.DrawLine($penGrid, $x, $H, ($x + $H), 0)
    }

    # Soft radial glow behind the badge.
    $badgeSize = [int]($W * 0.42)
    $badgeX = [int](($W - $badgeSize) / 2)
    $badgeY = [int]($H * 0.15)
    $glowR = [int]($badgeSize * 1.5)
    $glowRect = New-Object System.Drawing.Rectangle(
        [int]($W/2 - $glowR/2), [int]($badgeY + $badgeSize/2 - $glowR/2), $glowR, $glowR)
    $glowPath = New-Object System.Drawing.Drawing2D.GraphicsPath
    $glowPath.AddEllipse($glowRect)
    $glow = New-Object System.Drawing.Drawing2D.PathGradientBrush($glowPath)
    $glow.CenterColor    = [System.Drawing.Color]::FromArgb(120, 42, 112, 255)
    $glow.SurroundColors = @([System.Drawing.Color]::FromArgb(0, 42, 112, 255))
    $g.FillPath($glow, $glowPath)

    # Badge.
    $badge = New-BadgeBitmap $badgeSize
    $g.DrawImage($badge, $badgeX, $badgeY, $badgeSize, $badgeSize)
    $badge.Dispose()

    # Wordmark.
    $wordFont = New-Object System.Drawing.Font('Segoe UI', [single]($W * 0.135), [System.Drawing.FontStyle]::Bold, [System.Drawing.GraphicsUnit]::Pixel)
    $fmt = New-Object System.Drawing.StringFormat
    $fmt.Alignment = [System.Drawing.StringAlignment]::Center
    $wordRect = New-Object System.Drawing.RectangleF(0, [single]($badgeY + $badgeSize + $H * 0.03), $W, [single]($H * 0.12))
    $g.DrawString('Raeen', $wordFont, (New-Object System.Drawing.SolidBrush($White)), $wordRect, $fmt)

    # Cyan divider.
    $divW = [single]($W * 0.30)
    $g.FillRectangle((New-Object System.Drawing.SolidBrush($Cyan)),
        [single](($W - $divW)/2), [single]($badgeY + $badgeSize + $H * 0.135), $divW, [single]([Math]::Max(2.0, $H*0.004)))

    # Tagline.
    $tagFont = New-Object System.Drawing.Font('Segoe UI', [single]($W * 0.042), [System.Drawing.FontStyle]::Regular, [System.Drawing.GraphicsUnit]::Pixel)
    $tagRect = New-Object System.Drawing.RectangleF(0, [single]($badgeY + $badgeSize + $H * 0.155), $W, [single]($H * 0.1))
    $g.DrawString('PS5 Compatibility Layer', $tagFont, (New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(190, 200, 214, 255))), $tagRect, $fmt)

    $bg.Dispose(); $penGrid.Dispose(); $glow.Dispose(); $glowPath.Dispose()
    $wordFont.Dispose(); $tagFont.Dispose(); $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Bmp)
    $bmp.Dispose()
}

# --------------------------------------------------------- wizard header ----
# The badge on a white field, for the inner-page header (top-right). 24bpp.
function Write-WizardSmall([string]$path, [int]$W, [int]$H) {
    $bmp = New-Object System.Drawing.Bitmap($W, $H, [System.Drawing.Imaging.PixelFormat]::Format24bppRgb)
    $g = [System.Drawing.Graphics]::FromImage($bmp)
    $g.Clear($White)
    $g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
    $g.InterpolationMode = [System.Drawing.Drawing2D.InterpolationMode]::HighQualityBicubic
    $side = [int]([Math]::Min($W, $H) * 0.92)
    $badge = New-BadgeBitmap $side
    $g.DrawImage($badge, [int](($W - $side)/2), [int](($H - $side)/2), $side, $side)
    $badge.Dispose(); $g.Dispose()
    $bmp.Save($path, [System.Drawing.Imaging.ImageFormat]::Bmp)
    $bmp.Dispose()
}

# ---------------------------------------------------------------- run -------
$ico   = Join-Path $AssetsDir 'raeen.ico'
$large = Join-Path $AssetsDir 'wizard-large.bmp'
$small = Join-Path $AssetsDir 'wizard-small.bmp'

Write-Ico         $ico   @(16, 32, 48, 64, 128, 256)
Write-WizardLarge $large 328 628
Write-WizardSmall $small 165 174

Write-Host "Generated:"
Write-Host "  $ico"
Write-Host "  $large"
Write-Host "  $small"
