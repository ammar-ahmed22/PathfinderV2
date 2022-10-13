import * as React from "react";
import {
    useColorMode,
    useColorModeValue,
    IconButton,
    IconButtonProps,
    Switch,
    SwitchProps,
    HStack,
    StackProps,
    Icon,
    Text
} from "@chakra-ui/react";
import { FaMoon, FaSun } from "react-icons/fa";

type ColorModeSwitcherProps = Omit<StackProps, "aria-label">;

const ColorModeSwitcher: React.FC<ColorModeSwitcherProps> = (props) => {
    const { toggleColorMode } = useColorMode();
    const text = useColorModeValue("dark", "light");
    const SwitchIcon = useColorModeValue(FaMoon, FaSun);

    return (
        <HStack {...props} aria-label={`Switch to ${text} mode`} >
            <Text>☀️</Text>
            <Switch size="sm" onChange={() => toggleColorMode()}/>
            <Text>🌙</Text>
        </HStack>
    );
};

export default ColorModeSwitcher;
