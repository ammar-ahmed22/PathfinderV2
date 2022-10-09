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
        djikstra: "Djikstra's Algorithm",
        greedy: "Greedy Best First Search",
    };

    const handleChange = (val: string | string[]) => {
        if (typeof val === "string" && isAlgorithm(val)) {
            store.setSelectedAlgorithm(val);
        }
    };

    const filterAlgorithms = (algoNames: string[], selectedAlgo: string) => {
        return algoNames.filter( algoName => algoName !== selectedAlgo)
    }

    return (
        <Menu>
            <MenuButton as={Button} rightIcon={<ChevronDownIcon />}>
                {
                    store.selectedAlgorithm && algorithmMenuNames[store.selectedAlgorithm]
                }
            </MenuButton>
            <MenuList>
                <MenuOptionGroup
                    defaultValue="astar"
                    title="Algorithms"
                    type="radio"
                    onChange={handleChange}
                >
                    {filterAlgorithms(Object.keys(algorithmMenuNames), store.selectedAlgorithm).map((algoName) => {
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
