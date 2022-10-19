docker build - < DOCKERFILE -t example --build-arg SSH_PRIVATE_KEY="$(cat ~/.ssh/id_rsa_github_deploy)" # --no-cache

# docker run -i -d --runtime=nvidia --name CONTAINER_NAME \
#             -e DISPLAY=$DISPLAY -e QT_X11_NO_MITSHM=1 -e XAUTHORITY -e NVIDIA_DRIVER_CAPABILITIES=all\
#                     -v /tmp/.X11-unix:/tmp/.X11-unix \
#                             -v `pwd`/:/root/hza_home \
#                             -v `pwd`/cephfs:/cephfs \
#                                     gitlab-registry.nrp-nautilus.io/hzaskywalker/docker