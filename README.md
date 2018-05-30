# upper-body-recognisation

net.py
PAFs (Part Affinity Fields) are matrices that give information about the position and orientation of pairs. They come in couples: for each part we have a PAF in the ‘x’ direction and a PAF in the ‘y’ direction. There are 38 PAFs associated with each one of the pairs and indexed."net.py" code that runs this stage of the pipeline.

Non Maximum Suppression(nms.py)
Next step is detecting the parts in the image. Heatmaps are cool, but we should transform confidence into certainty if we want to move forward.

We need to extract parts locations out of a heatmap i.e.,we need to extract points out of a function


We apply a non-maximum suppression (NMS) algorithm to get those peaks.

Start in the first pixel of the heatmap.
Surround the pixel with a window of side 5 and find the maximum value in that area.
Substitute the value of the center pixel for that maximum
Slide the window one pixel and repeat these steps after we’ve covered the entire heatmap.
Compare the result with the original heatmap. Those pixels staying with same value are the peaks we are looking for. Suppress the other pixels setting them with a value of 0.
After all the process, the non-zero pixels denote the location of the part candidates.

To execute the nms algorithm, try running "nms.py".

LineIntergral.py

This is where PAFs enter the pipeline. We will compute the line integral along the segment connecting each couple of part candidates, over the corresponding PAFs (x and y) for that pair.The line integral will give each connection a score, that will be saved in a weighted bipartite graph and will allow us to solve the assignment problem.Try running the "lineintegral.py".

Assignment.py
The weighted bipartite graph shows all possible connections between candidates of two parts, and holds a score for every connection. The mission now is to find the connections that maximize the total score, that is, solving the assignment problem.

There are plenty of good solutions to this problem, but we are going to pick the most intuitive one:

Sort each possible connection by its score.
The connection with the highest score is indeed a final connection.
Move to next possible connection. If no parts of this connection have been assigned to a final connection before, this is a final connection.
Repeat the step above until we are done.Run the code "assignment.py".


Merging.py
The final step is to transform these detected connections into the final skeletons.
If humans H1 and H2 share a part index with the same coordinates, they are sharing the same part! H1 and H2 are, therefore, the same humans. So we merge both sets into H1 and remove H2. After all the merging is done, we finally describe a human as a set of parts.Try running "merging.py" code.


Output:
Finally what you get is a collection of human sets, where each human is a set of parts, where each part contains its index, its relative coordinates and its score.

run python inference.py -- imgpath /path/to/your/img to test












