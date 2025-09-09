import torch
import torch.nn.functional as F

def contrastive_multiview_loss(z, temp=0.1):
   """
   z: [N, P] = proyecciones normalizadas de TODOS los agentes en el MISMO paso t
      (si quieres usar más pasos, concatena y arma la máscara de positivos).
   Pérdida InfoNCE con multi-positivos: cada ejemplo tiene N-1 positivos (los otros agentes).
   """
   N = z.size(0)
   z = F.normalize(z, dim=-1)
   sim = z @ z.t() / temp                     # [N, N]
   mask_eye = torch.eye(N, device=z.device, dtype=torch.bool)
   sim = sim.masked_fill(mask_eye, -1e9)      # sin self-positives

   # Objetivos "suaves": uniforme sobre N-1 positivos
   target = torch.full((N, N), 0.0, device=z.device)
   target[~mask_eye] = 1.0 / (N - 1)

   logp = sim.log_softmax(dim=1)
   loss = -(target * logp).sum(dim=1).mean()
   return loss
