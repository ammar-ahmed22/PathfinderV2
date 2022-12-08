import React, { useEffect, useContext } from "react";
import {
    Menu,
    MenuButton,
    MenuList,
    Button,
    MenuOptionGroup,
    MenuItemOption,
} from "@chakra-ui/react";

// Icons
import { ChevronDownIcon } from "@chakra-ui/icons";

// Utils
import { isAlgorithm } from "../utils/types";

// Store
import { StoreContext } from "../Store";
import { StoreContextType } from "../@types/Store";

export const algorithmMenuNames: Record<string, string> = {
    astar: "A* Search",
    djikstra: "Djikstra's",
    greedy: "Greedy Best First",
};

const AlgorithmMenu: React.FC = () => {
    const store = useContext(StoreContext) as StoreContextType;

    useEffect(() => {
        store.setSelectedAlgorithm("astar");
        // eslint-disable-next-line
    }, []);

    const handleChange = (val: string | string[]) => {
        if (typeof val === "string" && isAlgorithm(val)) {
            store.setSelectedAlgorithm(val);
        }
    };

    return (
        <Menu matchWidth>
            <MenuButton
                as={Button}
                rightIcon={<ChevronDownIcon />}
                variant="brandPurple"
                size="sm"
            >
                Algorithm
            </MenuButton>
            <MenuList>
                <MenuOptionGroup
                    defaultValue="astar"
                    title="Algorithms"
                    type="radio"
                    onChange={handleChange}
                >
                    {Object.keys(algorithmMenuNames).map((algoName) => {
                        return (
                            <MenuItemOption value={algoName}>
                                {algorithmMenuNames[algoName]}
                            </MenuItemOption>
                        );
                    })}
                </MenuOptionGroup>
            </MenuList>
        </Menu>
    );
};

export default AlgorithmMenu;
