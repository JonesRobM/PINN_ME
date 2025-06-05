class PINNTrainer:
    def __init__(self, model, optimizer, physics_fn, bc_fn, config):
        self.model      = model
        self.optimizer  = optimizer
        self.physics_fn = physics_fn
        self.bc_fn      = bc_fn
        self.config     = config

    def train(self, collocation_loader, boundary_loader, epsilon_fn):
        for epoch in range(self.config['num_epochs']):
            self.model.train()
            # 1. Sample one batch of collocation points
            pts_coll = next(iter(collocation_loader)).to(self.config['device'])
            # 2. Sample one batch of BC points
            pts_bc = next(iter(boundary_loader)).to(self.config['device'])
            # 3. Zero gradients
            self.optimizer.zero_grad()
            # 4. Compute physics loss
            res = self.physics_fn(self.model, pts_coll, epsilon_fn,
                                   self.config['μ'], self.config['ω'])
            loss_phys = torch.mean(res**2)
            # 5. Compute BC loss
            loss_bc = self.bc_fn(self.model, pts_bc)
            # 6. Weighted total loss
            loss_total = self.config['w_phys']*loss_phys + self.config['w_bc']*loss_bc
            # 7. Backprop and step
            loss_total.backward()
            self.optimizer.step()
            # 8. (Optionally) log to TensorBoard
            ...
            # 9. Save checkpoint every N epochs
            if (epoch+1) % self.config['save_every'] == 0:
                save_checkpoint(self.model, self.optimizer, epoch,
                                f"outputs/checkpoints/pinn_epoch_{epoch+1:05d}.pt")
