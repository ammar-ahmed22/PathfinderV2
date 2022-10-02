import { extendTheme } from "@chakra-ui/react";
import { mode } from "@chakra-ui/theme-tools";
import type { StyleFunctionProps } from '@chakra-ui/styled-system'


const colors = {
  brand: {
    blue: "#667eea",
    purple: "#764ba2"
  }
}

const config = {
  cssVarPrefix: "pf"
}

const styles = {
  global: (props: StyleFunctionProps) => ({
    body: {
      fontFamily: "Noto Sans",
      bg: mode("gray.100", "gray.800")(props)
    }
  })
}

const shadows = {
  panel: "rgba(0, 0, 0, 0.1) 0px 4px 12px"
}

const components = {
  Heading: {
    variants: {
      gradient: {
        bgGradient: "linear(to-l, brand.blue, brand.purple)",
        bgClip: "text",
        fontWeight: "black"
      }
    }
  }
}

export default extendTheme({ colors, styles, config, shadows, components })
