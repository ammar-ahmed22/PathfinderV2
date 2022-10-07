import { QueueNode } from "../../@types/helpers/queue/MinPriorityQueue";

export default class MinPriorityQueue<T> {
    // Binary tree represented as array
    public heap: QueueNode<T>[] = [];

    // public methods
    public size = (): number => this.heap.length;
    public isEmpty = (): boolean => this.size() === 0;
    public peek = (): T | null => (this.isEmpty() ? null : this.heap[0].value);

    // methods to access values in the tree
    private parent = (i: number): number => Math.floor((i - 1) / 2);
    private left = (i: number): number => 2 * i + 1;
    private right = (i: number): number => 2 * i + 2;
    private hasLeft = (i: number): boolean => this.left(i) < this.size();
    private hasRight = (i: number): boolean => this.right(i) < this.size();
    private swap = (a: number, b: number): void => {
        const tmp = this.heap[a];
        this.heap[a] = this.heap[b];
        this.heap[b] = tmp;
    };

    public includes = (item: T): boolean => {
        for (let i = 0; i < this.size(); i++) {
            const curr = this.heap[i];
            if (curr.value === item) return true;
        }

        return false;
    };

    public insert = (item: T, priority: number) => {
        this.heap.push({ key: priority, value: item });

        let i = this.size() - 1;
        while (i > 0) {
            const p = this.parent(i);

            if (this.heap[p].key < this.heap[i].key) break;

            this.swap(i, p);

            i = p;
        }
    };

    public pop = (): T | undefined => {
        if (this.isEmpty()) return undefined;

        this.swap(0, this.size() - 1);

        const item: QueueNode<T> | undefined = this.heap.pop();

        let current: number = 0;

        // iterate while there are children
        while (this.hasLeft(current)) {
            // Find smaller child
            let smallerChild: number = this.left(current);
            if (
                this.hasRight(current) &&
                this.heap[this.right(current)].key <
                    this.heap[this.left(current)].key
            ) {
                smallerChild = this.right(current);
            }

            // If everything in order, break
            if (this.heap[smallerChild].key > this.heap[current].key) break;

            // Otherwise, swap and return to parent
            this.swap(current, smallerChild);
            current = smallerChild;
        }

        return item?.value;
    };
}
