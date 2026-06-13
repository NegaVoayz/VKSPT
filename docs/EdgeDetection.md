# Question



I'm solving a hard problem in graphics.

The problem is:

I'm doing photon mapping, but struggles with fine edges and faint region. If I set the radius large, the edges blur; If I set the radius tight, the faint regions go discrete.

My photon structure is `half spectrum[16]`,

and I came up with an idea:

How about calculate the distribution? I will set 2 constants `MaxPhotonCount` and `MaxGatherRadius`

Then do the following job:

text



```
let radius be InitialRadius;
let photons be gather(radius);
if (photons.total_photon_cnt >= MaxPhotonCount)
    calculate_and_return();

while(true) :
	new_radius = radius+delta;
    new_photons = gather(new_radius);

    if (EdgeDetected(new_photons))
        calculate_and_return();
    if (new_radius >= MaxGatherRadius)
        calculate_and_return();
    if (accum.total_photon_cnt >= MaxPhotonCount)
        calculate_and_return();
    radius = new_radius;
    photons = new_photons;
```

The edge detection should be able to detect at least these cases:

1. Normal edge (black | white)
2. Rainbow edge (red | blue)
3. Hybrid edge (pink | cyan)
4. Fake spectral edge (blue & red mixed evenly)
5. Faint edge (few photons | no photons)
6. Faint plain (few photons | few photons)



# Answer



This is a classic and notoriously difficult problem in Density Estimation and Photon Mapping: the **Bias vs. Variance tradeoff**. A large radius gives you high bias (blurry edges, light leaking), while a small radius gives you high variance (noisy, splotchy faint regions).

Your intuition to make the radius **adaptive** based on the local statistical distribution of the photons is exactly the right path.

However, we need to solve two things:

1. **The Edge Detection Math:** How to reliably identify edges across 16 spectral channels without false-flagging random noise.
2. **Algorithm Efficiency:** Iteratively calling `gather()` in a tree or hash grid will destroy your render times.

Here is a robust mathematical approach to solve this, tailored to your 16-channel spectral renderer.

------

### Part 1: The "Spectral-Spatial Dipole" Edge Detector

To detect all 6 of your cases, relying on photon *count* or *total energy* isn't enough. You need to look at the **spatial distribution of the energy** relative to the shading point.

We can do this by calculating the **Center of Mass (or Dipole Moment)** of the photons for *each of the 16 spectral channels individually*.

If you are shading a flat uniform plain, the photons surround the shading point evenly, and the center of mass for all 16 channels will be exactly at the shading point (distance = 0).
If there is an edge, the center of mass for the affected spectral channels will shift strongly away from the shading point.

#### The Math:

For a given search radius R*R*, for each spectral channel k*k* (from 0 to 15):

1. Calculate the center of mass vector D⃗k*D**k*:
   $$ \vec{D}*k = \frac{\sum (Photon.spectrum[k] \times \vec{V}*{photon})}{\sum Photon.spectrum[k]} $$
   *(Where V⃗photon\*V**p**h**o**t**o**n\*​ is the vector from the shading point to the photon).*
2. Normalize the magnitude by the radius to get a score between 0.0 and 1.0:
   $$ Score_k = \frac{|\vec{D}_k|}{R} $$

#### How this solves your 6 cases:

1. **Normal edge (black|white):** All 16 channels have photons on one side, none on the other. All D⃗k*D**k* point in the same direction. Score*S**cor**e* is high. **(Detected)**
2. **Rainbow edge (red|blue):** Red channels yield a D⃗red*D**r**e**d* pointing left. Blue channels yield D⃗blue*D**b**l**u**e* pointing right. The *total* energy might be centered, but per-channel scores will be high. **(Detected)**
3. **Hybrid edge (pink|cyan):** Same as above, the vectors for mixed channels will pull in opposing directions. **(Detected)**
4. **Fake spectral edge:** If red and blue are evenly mixed everywhere, D⃗red*D**r**e**d* and D⃗blue*D**b**l**u**e* are both 00. If they suddenly separate, the vectors diverge. **(Detected)**
5. **Faint edge (few|none):** Even with few photons, if they are all on one side, the center of mass shifts towards them. Score*S**cor**e* is high. **(Detected)**
6. **Faint plain (few|few):** *Here is the danger.* Because there are few photons, randomness might place them slightly off-center, causing a false positive. We fix this using a **statistical noise threshold**.

### Part 2: Filtering the Faint Plain (Poisson Noise)

In a uniform faint plain, photons arrive randomly (Poisson distribution). The statistical variance of their center of mass is proportional to 1Nk*N**k*1 (where Nk*N**k* is the effective number of photons contributing to channel k*k*).

To prevent a faint plain from being detected as an edge, your threshold for detecting an edge must adapt to the photon count:

$$ Threshold_k = \max\left( MinEdgeScore, \frac{Confidence}{\sqrt{N_k}} \right) $$

If Scorek>Thresholdk*S**cor**e**k*>*T**h**r**es**h**o**l**d**k*, you have hit a real edge, not just random noise.

------

### Part 3: Optimized Algorithm Structure

Calling `gather()` inside a `while` loop is highly inefficient. Instead, do **one maximum gather query** up front, sort the photons by distance, and simulate the expanding radius by iterating through the array.

Here is the pseudo-code for your complete solution:

C++



```
// Tuning constants
const float MinEdgeScore = 0.15f; // Theoretical max for a sharp edge is ~0.375
const float Confidence = 1.5f;    // Higher = more blur in faint plains, less false edges

void CalculateRadiance(ShadingPoint pt) {
    // 1. Gather ALL photons up to maximum limits ONCE.
    // This is vastly faster than querying the KD-Tree/Grid multiple times.
    PhotonArray photons = gather_max(pt.position, MaxGatherRadius, MaxPhotonCount);
    
    // Sort photons by distance to the shading point
    sort_by_distance(photons, pt.position);

    float current_radius = InitialRadius;
    int photon_index = 0;
    
    // Running accumulators
    float total_energy[16] = {0};
    vec3  dipole_moment[16] = {vec3(0,0,0)};
    int   photon_count = 0;

    while (photon_index < photons.size()) {
        Photon p = photons[photon_index];
        
        // If the next photon is outside our current simulation radius, 
        // we pause and evaluate the edge detection.
        if (p.distance > current_radius) {
            
            // --- EDGE DETECTION ---
            bool edge_detected = false;
            for (int k = 0; k < 16; k++) {
                if (total_energy[k] > 0.0001f) {
                    float score = length(dipole_moment[k]) / (current_radius * total_energy[k]);
                    
                    // Effective photon count for this channel 
                    // (prevents noise in channels with very little energy)
                    float Nk = photon_count * (p.spectrum[k] / max_channel(p.spectrum)); 
                    float noise_threshold = Confidence / sqrt(max(1.0f, Nk));
                    
                    if (score > max(MinEdgeScore, noise_threshold)) {
                        edge_detected = true;
                        break;
                    }
                }
            }

            if (edge_detected || photon_count >= MaxPhotonCount) {
                break; // Stop expanding!
            }
            
            // Expand radius
            current_radius += delta; 
        }

        // Accumulate photon into our running stats
        vec3 offset = p.position - pt.position;
        for (int k = 0; k < 16; k++) {
            total_energy[k] += p.spectrum[k];
            dipole_moment[k] += offset * p.spectrum[k];
        }
        
        photon_count++;
        photon_index++;
    }

    // current_radius is now your optimal density estimation radius!
    // photon_index is the number of photons inside that radius.
    return calculate_final_radiance(photons, photon_index, current_radius);
}
```

### Why this works so beautifully:

1. **Performance:** You only touch the spatial acceleration structure (KD-Tree/BVH) once.
2. **Spectral awareness:** Because it checks all 16 `half` channels independently, a sudden shift in *color* triggers the edge detector just as reliably as a sudden shift in *brightness*.
3. **Statistically sound:** By using 1N*N*1, the algorithm "knows" that 5 photons in a faint area will look asymmetrical just by pure chance, and it ignores it, allowing the radius to keep expanding until the faint area smoothes out.