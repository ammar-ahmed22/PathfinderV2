import React, { useEffect, useContext } from "react";
import {
    Menu,
    MenuButton,
    MenuList,
    MenuItem,
    Button,
    MenuOptionGroup,
    MenuItemOption,
} from "@chakra-ui/react";
import { ChevronDownIcon } from "@chakra-ui/icons";
import { StoreContext } from "../Store";

import { isAlgorithm } from "../utils/types";
import { StoreContextType } from "../@types/Store";

const AlgorithmMenu: React.FC = () => {
    const store = useContext(StoreContext) as StoreContextType;

    useEffect(() => {
        store.setSelectedAlgorithm("astar");
    }, []);

    const algorithmMenuNames: Record<string, string> = {
        astar: "A* Search",
        djikstra: "Djikstra's",
        greedy: "Greedy Best First",
    };

    const handleChange = (val: string | string[]) => {
        if (typeof val === "string" && isAlgorithm(val)) {
            store.setSelectedAlgorithm(val);
        }
    };

    const filterAlgorithms = (algoNames: string[], selectedAlgo: string) => {
        return algoNames.filter((algoName) => algoName !== selectedAlgo);
    };

    return (
        <Menu matchWidth >
            <MenuButton as={Button} rightIcon={<ChevronDownIcon />} variant="brandPurple" size="sm" >
                    Select Algorithm
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
