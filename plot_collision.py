import numpy as np
import matplotlib.pyplot as plt
from collision import SailingCollisionAvoidance, State, Action
from tqdm import tqdm

def plot_learned_policy():
    # Initialize system and learn policy
    system = SailingCollisionAvoidance()
    system.optimize_policy(num_iterations=100)
    
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
    fig.suptitle('Sailing Collision Avoidance Policy', fontsize=14)
    
    # Generate meshgrid for plotting
    ttc = system.ttc_points
    prob = system.prob_points
    TTC, PROB = np.meshgrid(ttc, prob)
    
    # Get actions for each priority case
    actions_priority = np.zeros_like(TTC)
    actions_no_priority = np.zeros_like(TTC)
    values_priority = np.zeros_like(TTC)
    values_no_priority = np.zeros_like(TTC)
    
    print("Calculating policy grid...")
    for i in range(len(ttc)):
        for j in range(len(prob)):
            # With priority
            state = State(time_to_collision=ttc[i], 
                        collision_prob=prob[j], 
                        has_priority=True)
            actions_priority[j,i] = system.get_action(state).value
            values_priority[j,i] = system.values[i,j,1]
            
            # Without priority
            state = State(time_to_collision=ttc[i], 
                        collision_prob=prob[j], 
                        has_priority=False)
            actions_no_priority[j,i] = system.get_action(state).value
            values_no_priority[j,i] = system.values[i,j,0]
    
    # Plot for priority case
    im1 = ax1.pcolor(TTC, PROB, actions_priority, cmap='RdYlBu', shading='auto')
    cs1 = ax1.contour(TTC, PROB, values_priority, colors='k', alpha=0.3)
    ax1.clabel(cs1, inline=1, fontsize=8)
    ax1.set_title('With Priority')
    ax1.set_xlabel('Time to Collision (s)')
    ax1.set_ylabel('Collision Probability')
    
    # Plot for no priority case
    im2 = ax2.pcolor(TTC, PROB, actions_no_priority, cmap='RdYlBu', shading='auto')
    cs2 = ax2.contour(TTC, PROB, values_no_priority, colors='k', alpha=0.3)
    ax2.clabel(cs2, inline=1, fontsize=8)
    ax2.set_title('Without Priority')
    ax2.set_xlabel('Time to Collision (s)')
    
    # Add colorbar
    cbar = fig.colorbar(im1, ax=[ax1, ax2])
    cbar.set_ticks([0.25, 0.75])
    cbar.set_ticklabels(['Continue', 'Tack'])
    
    # Add safety time threshold line
    for ax in [ax1, ax2]:
        ax.axvline(x=system.safety_time, color='r', linestyle='--', alpha=0.5)
        ax.text(system.safety_time + 1, 0.1, 'Safety\nThreshold', 
                rotation=90, color='r', alpha=0.5)
    
    # Add annotations explaining key regions
    def annotate_regions(ax):
        # High risk region
        ax.annotate('High Risk\nRegion', xy=(5, 0.8), xytext=(15, 0.9),
                   arrowprops=dict(facecolor='black', shrink=0.05), alpha=0.5)
        # Safe region
        ax.annotate('Safe\nRegion', xy=(50, 0.2), xytext=(40, 0.1),
                   arrowprops=dict(facecolor='black', shrink=0.05), alpha=0.5)
    
    annotate_regions(ax1)
    annotate_regions(ax2)
    
    plt.tight_layout()
    plt.show()
    
    # Print some policy analysis
    print("\nPolicy Analysis:")
    
    # Analyze earliest tacking points
    def find_earliest_tack(actions, ttc, prob):
        tack_mask = actions > 0.5
        earliest_tacks = []
        for p_idx in range(len(prob)):
            ttc_indices = np.where(tack_mask[p_idx,:])[0]
            if len(ttc_indices) > 0:
                earliest_tacks.append((prob[p_idx], ttc[ttc_indices[-1]]))
        return earliest_tacks
    
    priority_tacks = find_earliest_tack(actions_priority, ttc, prob)
    no_priority_tacks = find_earliest_tack(actions_no_priority, ttc, prob)
    
    print("\nWith Priority:")
    for prob_val, ttc_val in priority_tacks:
        print(f"At collision probability {prob_val:.2f}, earliest tack at {ttc_val:.1f}s")
        
    print("\nWithout Priority:")
    for prob_val, ttc_val in no_priority_tacks:
        print(f"At collision probability {prob_val:.2f}, earliest tack at {ttc_val:.1f}s")

if __name__ == '__main__':
    plot_learned_policy()