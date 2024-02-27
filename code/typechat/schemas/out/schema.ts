// The following is a schema definition for assigning a single likelihood (value) to each node (key).

export interface NodeLikelihoods {
    ellip0: number; // put here only the likelihood between 0 and 1 that the robot should grasp ellip0 and nothing else.
    ellip1: number; // put here only the likelihood between 0 and 1 that the robot should grasp ellip1 and nothing else.
}