def smoothed_z (z, L):

    lam = 1-2/(L+1)

    smoothed = z

    for i in range(len(z)):

        for j in range(1, len(z[1])):
            print(j-1)
            smoothed[i][j] = lam * smoothed[i][j-1] + (1-lam) * z[i][j]

    return smoothed


j = [[5,5,6,7,8,1,2,9],[2,3,4,1,5,2,4,8]]
l = 16

s = smoothed_z(j, l)

print(s)