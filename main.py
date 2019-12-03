from Channel_coding import *

epsilon = 0.05
Capacity = capacity(0.4)
print(Capacity)

# settings done 
theta = 0.4 # input distribution
r = 0.01
n = 600
########################channel#################################
p = 0.4 #flipover prob 
p_y_x = [[1-p, p], [p, 1-p]]
p_x = [1-theta, theta]

codebook = generate_codebook(theta =theta, n=n, r=r)
len(codebook)

p_joint = get_joint_pdf(p_y_x, p_x)
p_y = get_p_y(p_y_x, p_x)

print(p_x)
print(p_y_x)
print(p_joint)
print(p_y)

H_x = get_H_x(p_x)
H_y = get_H_y(p_y)
H_x_y = get_H_x_y(p_joint)


print(H_x)
print(H_y)
print(H_x_y)
print("Mutual Information : {}".format(H_x+H_y-H_x_y))



E=0
for i in tqdm(range((len(codebook)))):
    y_out= vector_channel(codebook[i], p)
    w_hat =  decoder(codebook,y_out, H_x, H_y, H_x_y, p_x, p_y, p_joint)
    if len(w_hat)==1:
        if w_hat[0]==i:
            E+=0
        else:
            E+=1
    else:
        E+=1

print("Average errpr over the transmission of codebook : {}".format(E/len(codebook)))


