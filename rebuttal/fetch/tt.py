if self.incremental_reward:
    self.prev_dist = np.linalg.norm(self.obs['achieved_goal'][self.obj_id * 3:self.obj_id * 3 + 3]
                                    - self.env.goal[self.obj_id * 3:self.obj_id * 3 + 3], self.norm)
    assert np.allclose(
        self.prev_dist,
        self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)[self.obj_id]
    )
    self.prev_contact = self.contact_dist()

self.obs, r, _, info = self.env.step(action)

# print('prev achieved', self.obs['achieved_goal'][self.obj_id * 3:(self.obj_id+1)*3])

subgoal_dists = self.subgoal_distances(self.obs['achieved_goal'], self.env.goal)
contact_dist = np.linalg.norm(self.obs['achieved_goal'][self.obj_id * 3:self.obj_id * 3 + 3]
                              - self.env.goal[self.obj_id * 3:self.obj_id * 3 + 3], self.norm)
assert np.allclose(subgoal_dists[self.obj_id], contact_dist)
