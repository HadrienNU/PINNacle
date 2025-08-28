import argparse
import time
import os
from trainer import Trainer

os.environ["DDEBACKEND"] = "pytorch"

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import matplotlib as mpl
from skfem.visuals.matplotlib import draw
import torch
import deepxde as dde
from scipy import interpolate
from src.model.laaf import DNN_GAAF, DNN_LAAF
from src.model.kan import KAN, build_splines_layers
from src.model.kan_utils.utils import plot
from src.optimizer import MultiAdam, LR_Adaptor, LR_Adaptor_NTK, Adam_LBFGS
from src.pde.poisson import Poisson2D_Classic, PoissonBoltzmann2D, Poisson_Ritz
from src.pde.burgers import Burgers1D
from src.pde.simple_test import KAN_Test, DeepRitz_Test
from src.pde.electromag import Magnetism_2D, Electric_2D, Magnetism_Ritz, Electric_Ritz
from src.utils.args import parse_hidden_layers, parse_loss_weight
from src.utils.callbacks import TesterCallback, PlotCallback, LossCallback
from src.utils.rar import rar_wrapper

pde_config = Burgers1D

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PINNBench trainer')
    parser.add_argument('--name', type=str, default="benchmark")
    parser.add_argument('--device', type=str, default="0")  # set to "cpu" enables cpu training
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--hidden-layers', type=str, default="100*5")
    parser.add_argument('--loss-weight', type=str, default="")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--iter', type=int, default=20000)
    parser.add_argument('--log-every', type=int, default=100)
    parser.add_argument('--plot-every', type=int, default=2000)
    parser.add_argument('--repeat', type=int, default=1)
    parser.add_argument('--method', type=str, default="adam")

    command_args = parser.parse_args()

    seed = command_args.seed
    if seed is not None:
        dde.config.set_random_seed(seed)
    date_str = time.strftime('%m.%d-%H.%M.%S', time.localtime())
    trainer = Trainer(f"{date_str}-{command_args.name}", command_args.device)

    activation_storage = []
    relu_hooks = []
    def get_relu_hook():
        def hook(module, input, output):
            # Store binary ReLU activation pattern (1 = active, 0 = inactive)
            activation_storage.append((output > 0).int().cpu())
        return hook

    def get_model_dde():
        if "disk" in command_args.name:
            pde = pde_config(form="disk")
        elif "ellipse" in command_args.name:
            pde = pde_config(form="ellipse")
        elif "polygon" in command_args.name:
            pde = pde_config(form="polygon")
        else:
            pde = pde_config()
        
        # pde.training_points()
        if command_args.method == "gepinn":
            pde.use_gepinn()

        architecture = "mlp"
        net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "tanh", "Glorot normal")
        if command_args.method == "laaf":
            net = DNN_LAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
        elif command_args.method == "gaaf":
            net = DNN_GAAF(len(parse_hidden_layers(command_args)) - 1, parse_hidden_layers(command_args)[0], pde.input_dim, pde.output_dim)
        elif command_args.method == "kan":
            net = KAN(build_splines_layers(
                [pde.input_dim, 5, pde.output_dim], 
                grid_size=10, 
                grid_alpha=0.02, 
                scale_basis=0.1, 
                auto_grid_update=True, 
                stop_grid_update_iter=(command_args.iter * 0.6)))
            architecture = "kan"
        elif command_args.method == "deepritz":
            net = dde.nn.FNN([pde.input_dim] + parse_hidden_layers(command_args) + [pde.output_dim], "relu", "Glorot normal")
            architecture = "deepritz"
        elif command_args.method == "kan-deepritz":
            net = KAN(build_splines_layers(
                [pde.input_dim, 5, pde.output_dim], 
                grid_size=10, 
                grid_alpha=0.02, 
                scale_basis=0.1, 
                auto_grid_update=True, 
                stop_grid_update_iter=(command_args.iter * 0.6),
                spline_order=1,
                sb_trainable=False,
                scale_base=0,
                base_activation=torch.nn.ReLU
                ))
            architecture = "kan-deepritz"
        net = net.float()

        loss_weights = parse_loss_weight(command_args)
        if loss_weights is None:
            loss_weights = np.ones(pde.num_loss)
        else:
            loss_weights = np.array(loss_weights)

        opt = torch.optim.Adam(net.parameters(), command_args.lr)
        if command_args.method == "multiadam":
            opt = MultiAdam(net.parameters(), lr=1e-3, betas=(0.99, 0.99), loss_group_idx=[pde.num_pde])
        elif command_args.method == "lra":
            opt = LR_Adaptor(opt, loss_weights, pde.num_pde)
        elif command_args.method == "ntk":
            opt = LR_Adaptor_NTK(opt, loss_weights, pde)
        elif command_args.method == "lbfgs":
            opt = Adam_LBFGS(net.parameters(), switch_epoch=5000, adam_param={'lr':command_args.lr})
        elif command_args.method == "kan":
            opt = Adam_LBFGS(net.parameters(), switch_epoch=0, adam_param={'lr':command_args.lr}, lbfgs_param={
                                                                                                            'lr':1, 
                                                                                                            'history_size':15, 
                                                                                                            'line_search_fn':"strong_wolfe", 
                                                                                                            'tolerance_grad':1e-32, 
                                                                                                            'tolerance_change':1e-32
                                                                                                        })

        exp_name = f"{date_str}-{command_args.name}"
        model = pde.create_model(net, architecture, exp_name)
        if architecture == "deepritz" or architecture == "kan-deepritz":
            model.compile(opt, loss_weights=loss_weights, loss="ritz")
        else:
            model.compile(opt, loss_weights=loss_weights)
        if command_args.method == "rar":
            model.train = rar_wrapper(pde, model, {"interval": 1000, "count": 1})
            
        return model

    trainer.add_task(
        get_model_dde, {
            "iterations": command_args.iter,
            "display_every": command_args.log_every,
            "callbacks": [
                TesterCallback(log_every=command_args.log_every),
                PlotCallback(log_every=command_args.plot_every, fast=True),
                LossCallback(verbose=True),
            ]
        }
    )

    trainer.setup(__file__, seed)
    trainer.set_repeat(command_args.repeat)
    trainer.train_all()
    trainer.summary()

    for model in trainer.trained_models:
        n_params = sum(p.numel() for p in model.net.parameters() if p.requires_grad)
        print("Trainable parameters:", n_params)

        if pde_config == Magnetism_2D or pde_config == Magnetism_Ritz:
            data = np.loadtxt(f"runs/{date_str}-{command_args.name}/0-0/model_output.txt", comments="#", delimiter=" ")
            if "disk" in command_args.name:
                pde = pde_config(form="disk")
            elif "ellipse" in command_args.name:
                pde = pde_config(form="ellipse")
            elif "polygon" in command_args.name:
                pde = pde_config(form="polygon")
            else:
                pde = pde_config()
            new_data = pde.geom.random_points(5000)
            #new_data_bound = pde.geom.random_boundary_points(1000)
            #new_data = np.vstack([new_data_in, new_data_bound])

            model.restore(f"runs/{date_str}-{command_args.name}/0-0/{command_args.iter}.pt")

            x, y, u, v = data[:, 0], data[:, 1], data[:, 2], data[:, 3]
            xy = data[:, 0:2]
            ref_uv = pde.ref_sol(new_data)
            u_ref, v_ref = ref_uv[:, 0], ref_uv[:, 1]
            output = model.predict(new_data)
            x_new, y_new = new_data[:, 0], new_data[:, 1]
            u_inference, v_inference = output[:, 0], output[:, 1]
            diff_u = u_inference - u_ref
            diff_v = v_inference - v_ref
            # Relative error magnitude
            ref_mag = np.sqrt(u_ref**2 + v_ref**2)
            diff_color = np.sqrt(diff_u**2 + diff_v**2) / (ref_mag + 1e-12)  # avoid division by 0
            norm_diff = mpl.colors.Normalize(vmin=0, vmax=1)  # 0–100% error


            color = np.sqrt((u)**2 + (v)**2)
            color_ref = np.sqrt((u_ref)**2 + (v_ref)**2)
            color_inference = np.sqrt((u_inference)**2 + (v_inference)**2)
            diff_color = np.sqrt(diff_u**2 + diff_v**2)

            plt.cla()
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # Create a normalization for shared scale
            vmin = min(color_ref.min(), color_inference.min())
            vmax = max(color_ref.max(), color_inference.max())
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis   # or any cmap you like

            # Reference
            q1 = axes[0].quiver(x_new, y_new, u_ref, v_ref, color_ref, cmap=cmap, norm=norm)
            axes[0].set_aspect("equal")
            axes[0].set_title("Reference Solution Vectors")

            # Inference
            q2 = axes[1].quiver(x_new, y_new, u_inference, v_inference, color_inference, cmap=cmap, norm=norm)
            axes[1].set_aspect("equal")
            axes[1].set_title("Model Inference Vectors")

            # Shared colorbar for reference & inference
            cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), ax=axes[0:2],
                                orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Magnitude")

            # Difference (independent color scale)
            q3 = axes[2].quiver(
                x_new, y_new, diff_u, diff_v, diff_color,
                cmap=plt.cm.viridis, norm=norm_diff
            )
            axes[2].set_aspect("equal")
            axes[2].set_title("Relative Error Vectors")
            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm_diff, cmap=plt.cm.viridis),
                ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04,
                label="Relative Error"
            )

            #plt.tight_layout()
            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/vectors", dpi=800)
            plt.close()
        
        elif pde_config == Electric_2D or pde_config == Electric_Ritz or pde_config == KAN_Test or pde_config == DeepRitz_Test or pde_config == Poisson2D_Classic:
            data = np.loadtxt(f"runs/{date_str}-{command_args.name}/0-0/model_output.txt", comments="#", delimiter=" ")
            if "disk" in command_args.name:
                pde = pde_config(form="disk")
            elif "ellipse" in command_args.name:
                pde = pde_config(form="ellipse")
            elif "polygon" in command_args.name:
                pde = pde_config(form="polygon")
            else:
                pde = pde_config()
            new_data = pde.geom.random_points(5000)

            model.restore(f"runs/{date_str}-{command_args.name}/0-0/{command_args.iter}.pt")

            x, y, o = data[:, 0], data[:, 1], data[:, 2]
            xy = data[:, 0:2]
            o_ref = pde.ref_sol(new_data)
            o_inference = model.predict(new_data)
            x_new, y_new = new_data[:, 0], new_data[:, 1]

            xx = np.linspace(np.min(x), np.max(x), 100)
            yy = np.linspace(np.min(y), np.max(y), 100)
            xx, yy = np.meshgrid(xx, yy)

            xx_new = np.linspace(np.min(x_new), np.max(x_new), 100)
            yy_new = np.linspace(np.min(y_new), np.max(y_new), 100)
            xx_new, yy_new = np.meshgrid(xx_new, yy_new)

            vals_ref = interpolate.griddata(np.array([x_new, y_new]).T, np.array(o_ref), (xx_new, yy_new), method='cubic')
            vals_0_ref = interpolate.griddata(np.array([x_new, y_new]).T, np.array(o_ref), (xx_new, yy_new), method='nearest')
            vals_ref[np.isnan(vals_ref)] = vals_0_ref[np.isnan(vals_ref)]
            vals_ref[~pde.geom.inside(np.stack((xx_new, yy_new), axis=2))] = np.nan
            vals_ref = vals_ref[::-1, :]

            vals = interpolate.griddata(np.array([x, y]).T, np.array(o), (xx, yy), method='cubic')
            vals_0 = interpolate.griddata(np.array([x, y]).T, np.array(o), (xx, yy), method='nearest')
            vals[np.isnan(vals)] = vals_0[np.isnan(vals)]
            vals[~pde.geom.inside(np.stack((xx, yy), axis=2))] = np.nan
            vals = vals[::-1, :]

            vals_inference = interpolate.griddata(np.array([x_new, y_new]).T, np.array(o_inference), (xx_new, yy_new), method='cubic')
            vals_0_inference = interpolate.griddata(np.array([x_new, y_new]).T, np.array(o_inference), (xx_new, yy_new), method='nearest')
            vals_inference[np.isnan(vals_inference)] = vals_0_inference[np.isnan(vals_inference)]
            vals_inference[~pde.geom.inside(np.stack((xx_new, yy_new), axis=2))] = np.nan
            vals_inference = vals_inference[::-1, :]

            vmin = min(np.nanmin(vals_ref), np.nanmin(vals_inference))
            vmax = max(np.nanmax(vals_ref), np.nanmax(vals_inference))
            norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            cmap = plt.cm.viridis

            plt.cla()
            fig, axes = plt.subplots(1, 3, figsize=(25, 5))

            # Reference heatmap
            im1 = axes[0].imshow(
                vals_ref,
                extent=[np.min(x_new), np.max(x_new), np.min(y_new), np.max(y_new)],
                aspect="auto", interpolation="bicubic", cmap=cmap, norm=norm
            )
            axes[0].set_xlabel("x")
            axes[0].set_ylabel("y")
            axes[0].set_title("Reference Solution Heatmap")
            axes[0].axis("equal")

            # Inference heatmap
            im2 = axes[1].imshow(
                vals_inference,
                extent=[np.min(x_new), np.max(x_new), np.min(y_new), np.max(y_new)],
                aspect="auto", interpolation="bicubic", cmap=cmap, norm=norm
            )
            axes[1].set_xlabel("x")
            axes[1].set_ylabel("y")
            axes[1].set_title("Inference Heatmap")
            axes[1].axis("equal")

            # Shared colorbar for ref & inference
            cbar = fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                ax=axes[0:2], orientation="vertical", fraction=0.046, pad=0.04
            )
            cbar.set_label("Magnitude")

            # -----------------------------
            # Error heatmap
            vals_diff = vals_inference - vals_ref # error
            vmin = np.nanmin(vals_diff)
            vmax = np.nanmax(vals_diff)
            norm_diff = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
            im3 = axes[2].imshow(
                vals_diff,
                extent=[np.min(x_new), np.max(x_new), np.min(y_new), np.max(y_new)],
                aspect="auto", interpolation="bicubic", cmap=plt.cm.viridis, norm=norm_diff
            )
            axes[2].set_xlabel("x")
            axes[2].set_ylabel("y")
            axes[2].set_title("Difference (Inference - Reference)")
            axes[2].axis("equal")

            fig.colorbar(
                mpl.cm.ScalarMappable(norm=norm_diff, cmap=plt.cm.viridis),
                ax=axes[2], orientation="vertical", fraction=0.046, pad=0.04,
                label="Difference Error"
            )

            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/heatmaps", dpi=800)
            plt.close()

            x_in = torch.from_numpy(xy).float().to("cuda").requires_grad_()
            u = model.net(x_in)
            pde_out = pde.pde(x_in, u)[0].cpu().detach().numpy()
            err = abs(pde_out)**2

            vals_err = interpolate.griddata(np.array([x, y]).T, np.array(err), (xx, yy), method='cubic')
            vals_0_err = interpolate.griddata(np.array([x, y]).T, np.array(err), (xx, yy), method='nearest')
            vals_err[np.isnan(vals_err)] = vals_0_err[np.isnan(vals_err)]
            vals_err[~pde.geom.inside(np.stack((xx, yy), axis=2))] = np.nan
            vals_err = vals_err[::-1, :]

            plt.cla()
            plt.figure()
            plt.imshow(vals_err, extent=[np.min(x), np.max(x), np.min(y), np.max(y)], aspect='auto', interpolation='bicubic')
            plt.colorbar(label='Charge Err')
            plt.xlabel("x")
            plt.ylabel("y")
            plt.axis("equal")
            plt.title("PDE Output Error")
            plt.tight_layout()
            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/pde_err", dpi=300)
            plt.close()      

        if command_args.method == "kan" or command_args.method == "kan-deepritz":
            data = np.loadtxt(f"runs/{date_str}-{command_args.name}/0-0/model_output.txt", comments="#", delimiter=" ")
            xy = data[:, 0:2]

            model.restore(f"runs/{date_str}-{command_args.name}/0-0/{command_args.iter}.pt")
            plot(model.net, title="KAN", tick=False, norm_alpha=True, beta=10)
            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/kan", dpi=300)

            pruned_model = model.net.prune(mode="auto")
            #pruned_model(torch.tensor(xy, dtype=torch.float32))
            plot(pruned_model, title="KAN after pruning", tick=False, norm_alpha=True, beta=10)
            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/kan_pruned", dpi=300)
            plt.close()

        if command_args.method == "deepritz" or command_args.method == "kan-deepritz":
            for module in model.net.modules():
                relu_hooks.append(module.register_forward_hook(get_relu_hook()))

            model.restore(f"runs/{date_str}-{command_args.name}/0-0/{command_args.iter}.pt")
            pde = pde_config()

            x_range = torch.linspace(pde.bbox[0], pde.bbox[1], 750)
            y_range = torch.linspace(pde.bbox[2], pde.bbox[3], 750)
            xx, yy = torch.meshgrid(x_range, y_range, indexing='ij')
            grid_points = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
            inside_mask = pde.geom.inside(grid_points.cpu().numpy())
            valid_points = grid_points[inside_mask]
            _ = model.predict(valid_points.cpu().numpy())

            # Combine all recorded ReLU layer activations into one binary vector per input
            print([act.shape for act in activation_storage])
            activation_patterns = torch.cat(activation_storage, dim=1)  # shape: [num_points, total_relus]
            activation_storage.clear()  # Clean up

            # Convert each activation pattern to a unique ID
            pattern_ids = activation_patterns.numpy().dot(1 << np.arange(activation_patterns.shape[1]))
            unique_ids, remapped_ids = np.unique(pattern_ids, return_inverse=True)
            print("Number of region", len(unique_ids))
            remapped_ids += 1  # So it starts from 1 instead of 0
            
            num_base_colors = 40
            base_cmap = cm.get_cmap('tab20', num_base_colors)

            # Map activation region IDs using modulo
            modulo_color = remapped_ids % num_base_colors

            if pde_config == Electric_Ritz:
                fem_mesh = pde.mesh
                plt.figure(figsize=(8, 6))
                draw(fem_mesh)
                plt.scatter(valid_points[:, 0].cpu().numpy(), valid_points[:, 1].cpu().numpy(), c=modulo_color, cmap=base_cmap, s=0.2)
                #plt.colorbar(label='Linear Region ID (Activation Pattern)')
                plt.title('Linear Regions via ReLU Activation Patterns (Hook-Based) and FEM Mesh')
                plt.xlabel('x1')
                plt.ylabel('x2')
                plt.tight_layout()
                plt.axis("equal")
                plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/activation_pattern_mesh", dpi=300)
                plt.close()  

            plt.figure(figsize=(8, 6))
            plt.scatter(valid_points[:, 0].cpu().numpy(), valid_points[:, 1].cpu().numpy(), c=modulo_color, cmap=base_cmap, s=0.2)
            #plt.colorbar(label='Linear Region ID (Activation Pattern)')
            plt.title('Linear Regions via ReLU Activation Patterns (Hook-Based)')
            plt.xlabel('x1')
            plt.ylabel('x2')
            plt.tight_layout()
            plt.axis("equal")
            plt.savefig(f"runs/{date_str}-{command_args.name}/0-0/activation_pattern", dpi=300)
            plt.close()  
