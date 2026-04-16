FROM python:3.12-slim

# Install dependencies
RUN apt-get update && \
    apt-get -y install curl git vim nano libcairo2-dev graphviz

# Copy the project files into the container
COPY . /app
WORKDIR /app

# Install all dependencies directly into system Python
# When EXPERIMENTAL=1, install stormpy and paynt from local wheels (experimental builds).
ARG EXPERIMENTAL=0
RUN if [ "$EXPERIMENTAL" = "1" ]; then \
        pip install --no-cache-dir . stormpy paynt pyscipopt && \
        pip install --no-cache-dir --no-deps --force-reinstall stormpy_wheel/*cp312*.whl && \
        pip install --no-cache-dir --no-deps --force-reinstall paynt_wheel/*cp312*.whl; \
    else \
        pip install --no-cache-dir . stormpy paynt pyscipopt; \
    fi

# create /root/.jupyter directory
RUN mkdir -p /root/.jupyter

# Create a random password for the Jupyter Lab
RUN PASSWORD=$(echo -n $(date +%s) | sha1sum | awk '{print $1}') && echo $PASSWORD > /root/jupyter_password.txt

# Set identity provider class to token based
# Set the token to the password
RUN echo "c.NotebookApp.token = '$(cat /root/jupyter_password.txt)'" >> /root/.jupyter/jupyter_notebook_config.py

RUN echo "echo -e '\033[44;37mWelcome to the stormvogel container!\033[0m'" >> /root/.bashrc
RUN echo "\033[34m         =======                               \n\
      =============                            \n\
     ===============                           \n\
    =================           =====          \n\
    ======%%%=========        ============     \n\
   =====================     ===============   \n\
  ==========================================   \n\
     ====================================      \n\
    ================================         \n\
    =============================            \n\
    ==========++===============              \n\
     ==========##===============#            \n\
      ==========###===========#              \n\
        ==========####=====##                \n\
        ===========######                  \n\
             ==   ==                       \n\
            ===   ==                       \n\
         ====   ====                       \n\
            ====                         \033[0m" > /root/bird.txt
RUN echo "cat /root/bird.txt" >> /root/.bashrc
RUN echo "echo -e '\033[44;37mRun this container with -p 8080:8080 to get access to the Jupyter Lab from your host computer.\033[0m'" >> /root/.bashrc
# Print the Jupyter Lab URL, including the password
RUN echo "echo -e '\033[44;37mJupyter Lab will be running at http://localhost:8080/?token=$(cat /root/jupyter_password.txt) in a minute or so.\033[0m'" >> /root/.bashrc
# Print how to restart this docker instance after leaving it
RUN echo "echo -e \"\033[44;37mTo restart this container, run docker start -i \$(hostname)\033[0m\"" >> /root/.bashrc

# Start a bash shell, but run Jupyter Lab inside Poetry in the background on port 8080
CMD ["bash", "-c", "setsid jupyter lab --ip 0.0.0.0 --port=8080 --no-browser --allow-root 0</dev/null > /app/jupyter_lab.log 2>&1 & exec bash"]
