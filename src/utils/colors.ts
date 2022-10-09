
export interface RGB {
  r: number,
  g: number,
  b: number
}

export const hexToRGB = (hexColor: string) : RGB | null => {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hexColor)

  return result ? {
    r: parseInt(result[1], 16),
    g: parseInt(result[2], 16),
    b: parseInt(result[3], 16)
  } : null
}

export const RGBtoHex = (color: RGB) : string => {
  const componentToHex = (c: number) : string => {
    const hex = c.toString(16);
    return hex.length === 1 ? "0" + hex : hex;
  }

  return `#${componentToHex(color.r)}${componentToHex(color.g)}${componentToHex(color.b)}`
}

export const lerp = (percent: number, startColor: RGB, endColor: RGB) : RGB => {
  return {
    r: Math.round(startColor.r + percent * (endColor.r - startColor.r)),
    g: Math.round(startColor.g + percent * (endColor.g - startColor.g)),
    b: Math.round(startColor.b + percent * (endColor.b - startColor.b)),
  }
}

interface createGradientOpts{
  values: number,
  startColor: RGB,
  endColor: RGB,
  output?: "hex" | "rgb"
}

export const createGradient = ({ values, startColor, endColor, output = "hex"} : createGradientOpts) : (string | RGB)[] => {
  const res : (string | RGB)[] = [];

  for (let i = 0; i < values; i++){
    const percent = (i + 1) / values;
    const color = lerp(percent, startColor, endColor);
    if (output === "hex"){
      res.push(RGBtoHex(color))
    } else {
      res.push(color)
    }
  }

  return res;
}