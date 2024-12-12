## Auto extending segments along predictions or other indicator data 

To automatically follow predicitons/other binary data from , you must first create a segment . It is critical that when this is created, that you initalize it by creating 3 consecutive nodes in Z and then hitting "X" on the middle one, and creating 2 nodes in x or y on either side of it. If you start your fragment by attempting to extrapolate your first node, you will not get good results. I typically create the first 6 in a "cross" pattern, then begin extending the ends of the crosses to form a more rectangular pattern. The same rules apply with autosegmenting as they do with typical khartes usage: 

- try and keep your segments relatively rectangular , parameterization gets very strange and you will encounter issues if you create very long triangle edges
- try your best to not place nodes right next to eachother. the algorithm attempts to space them with decent defaults, but it is good practice to still attempt to avoid this
- in tight wraps, if you find your nodes connecting strangely , you will want extend in the direction that is opposite of that plane -- so for z typically, if you try and go around a wrap and your points connect strangely, extend in x or y

### Keybinds : 
- Press (3) to extend your segment to the "left"
- Press (4) to extend your segment to the "right"
- Press (5) to delete a node
- Press (`) to reparamaterize

It is good practice to reparameterize after every extension, as this runs a global reparameterization and general mesh cleanup that will help you avoid issues later

### How it works and how to get the most out of it 

This auto-extrapolation works like this: 
1. taking the active data window, thresholding for some value (defaults to 50% of uint16, or aboout 32000) ,
2. compute connected components on this subvolume ,
3. find the label value that the currently selected node exists on , if any
4. evict this component from the mask, and return the positions of all the pixels in this component,
5. identify the pixel furthest from the node location (or the last pixel in the line)
6. fit a curve from the beginning node to the last node, through the rest of the component, and stop a little short of the data window bounds so that we can actually see the last node
7. Add points to the data window/triangle fragment surface utlizing the existing khartes functions/classes so we dont have to rewrite the wheel. to khartes, all these points are added manually by a user. there is no difference in how uvs/xyz are computed. 

Because connected components are only computed on the data window, we can use this to our advantage in a few ways. If our component is well defined and not merged with nearby surfaces, we can zoom quite far out and extrapolate a pretty large distance. Conversely, if the components are merged we can simply zoom in or pan to a differenct slice/location that is not merged, and extend from there. If our extension stops where a component ends, we simply place a new node manually on this new component, and rerun the extrapolation. 

it computes a local region of connected components, gets all the pixels, and drives a line through them using interpolation. some bugs , not fully fleshed out. one major bug is tracking cursors cause crashes i have yet to track down.

to use it , set the overlay as the base volume and the raw data as the overlay, as in this screenshot. you can then adjust the opacity of the overlay (the data volume) to see the predictions. if you'd prefer, you dont actually need to do this, and can view only the data volume and still utilize this extrapolation so long as the overlay is set to the base volume. 

![image](https://github.com/user-attachments/assets/f3f91f55-1c19-453d-a99a-bdbdddd8eead)
