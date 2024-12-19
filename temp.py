import numpy as np

output = np.random.rand(3549,25)
print(output.shape)

from line_profiler import profile


@profile
def getBoxesFromOutput(values):
        boxes = []

        NUM_COLORS = 8
        NUM_TAGS = 8

        indices = np.where(values[:, 8] > 0.85)
        values = values[indices]
        
        print(len(values))

        for element in values:
            x_1 = (element[0])
            y_1 = (element[1])
            x_2 = (element[2])
            y_2 = (element[3])
            x_3 = (element[4])
            y_3 = (element[5])
            x_4 = (element[6])
            y_4 = (element[7])

            confidence = element[8]

            color = np.argmax(element[9 : 9 + NUM_COLORS])
            tag = np.argmax(element[9 + NUM_COLORS : 9 + NUM_COLORS + NUM_TAGS])

        return boxes


from time import perf_counter_ns

# Warm up
ITERATIONS = 1
start = perf_counter_ns()
for i in range(ITERATIONS):
  getBoxesFromOutput(output)
end = perf_counter_ns()

print(f"Inference time: {(end - start) * 1.0 / ITERATIONS * 1e-6} ms")
