import { extendTheme } from "@chakra-ui/react";
import { mode } from "@chakra-ui/theme-tools";
import type { StyleFunctionProps } from "@chakra-ui/styled-system";

const colors = {
    brand: {
        blue: {
            100: "#E1E8FD",
            200: "#C3D1FC",
            300: "#A4B7F8",
            400: "#8BA0F2",
            500: "#667EEA",
            600: "#4A5FC9",
            700: "#3344A8",
            800: "#202D87",
            900: "#131D70",
        },
        purple: {
            100: "#F2DFFA",
            200: "#E2C0F5",
            300: "#C599E3",
            400: "#A276C7",
            500: "#764BA2",
            600: "#5C368B",
            700: "#442574",
            800: "#30175D",
            900: "#210E4D",
        },
    },
    path: {
        start: "#D38312",
        end: "#A83279",
    },
};

const config = {
    cssVarPrefix: "pf",
};

const styles = {
    global: (props: StyleFunctionProps) => ({
        body: {
            fontFamily: "Noto Sans",
            bg: mode("gray.100", "gray.800")(props),
        },
    }),
};

const shadows = {
    panel: "rgba(0, 0, 0, 0.1) 0px 4px 12px",
};

const components = {
    Heading: {
        variants: {
            gradient: {
                bgGradient: "linear(to-l, brand.blue.500, brand.purple.500)",
                bgClip: "text",
                fontWeight: "black",
            },
        },
    },
    Link: {
        variants: {
            icon: {
                _hover: {
                    borderBottomStyle: "solid",
                    borderBottomWidth: "1px",
                },
            },
        },
    },
    Button: {
        variants: {
            brandPurple: {
                bg: "brand.purple.500",
                color: "white",
                _hover: {
                    bg: "brand.purple.600"
                }
            }
        }
    }
};

export default extendTheme({ colors, styles, config, shadows, components });
