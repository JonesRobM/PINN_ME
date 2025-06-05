def dirichlet_loss(net, pts_bc):
    Ez_bc = net(pts_bc)
    return torch.mean(Ez_bc**2)

def neumann_loss(net, pts_bc, normals):
    """
    pts_bc: (N_bc,2), normals: (N_bc,2) unit normals at each bc point
    """
    pts_bc.requires_grad_(True)
    Ez_bc = net(pts_bc)                                 # (N_bc,1)
    grad_E = torch.autograd.grad(Ez_bc, pts_bc,
                                 torch.ones_like(Ez_bc),
                                 create_graph=True)[0]     # (N_bc,2)
    dE_dn = torch.sum(grad_E * normals, dim=1, keepdim=True)
    return torch.mean(dE_dn**2)
