# Continuous integration with Jenkins on Arch Linux

I am going to try to get Jenkins to do my automatic testing for this
project. I will be doing this on my UM machine that is running Arch
Linux.

## Sources 

Used for inspiration, not followed to the letter since some things
have changed since they were written

- [this gist](https://gist.github.com/misterbrownlee/3708738)
- [this page](http://cvuorinen.net/2013/06/installing-jenkins-ci-server-with-github-integration-for-a-php-project/)
- [this one to setup authentification](http://geeks.aretotally.in/basic-jenkins-setup-on-aws-with-github-authentication-for-scala-projects/)

## Install Jenkins
Install it and make it run in the background automatically
```{bash}
sudo pacman -S jenkins
sudo systemctl start jenkins
sudo systemctl enable jenkins
```
Check that it's running locally
```{bash}
curl -I localhost:8090
```

You may need to open a port if you configured UFW correctly.
```{bash}
sudo ufw allow 8090
```

Then setup an ssh tunnel so we can access it in a browser for the setup from another computer (don't actually do this)
```{bash}
ssh -qNf -L8090:localhost:8090 YOURUSERNAME@example.com
```


## Configure Jenkins
1. With browser, on the local machine, go to `http://localhost:8090` (we have an ssh tunnel
   setup to do this. We are basically tunneling to the Arch sever
   running jenkins) and follow the instructions there. Get the
   password from the Arch server by looking at
   `/var/lib/jenkins/secrets/initialAdminPassword`

2. Within the browser, select plugins to install.
   - [Git plugin](https://wiki.jenkins-ci.org/display/JENKINS/Git+Plugin)
   - [GitHub plugin](https://wiki.jenkins-ci.org/display/JENKINS/Github+Plugin)

3. Navigate to Manage Jenkins > Configure System. There, configure the
   Jenkins URL and sysadmin email address. Then restart Jenkins
   
4. Set the variables for Global "Config user.name" and "Global Config user.email"

5. `ssh` to the jenkins server and become the jenkins user and generate the ssh key
	```{bash}
	sudo -u jenkins bash
	cd $HOME
	ssh-keygen
	```	

6. Now add that public key to a Github account. This can be a specific
   one specially created for jenkins (e.g. dummy-jenkins) or just one
   of your own accounts. Also as the jenkins user, do
	```{bash}
	ssh -T git@github.com
	```

## Configure a new job
Follow these references (all pretty equivalent):
- [very good one](https://code.tutsplus.com/tutorials/setting-up-continuous-integration-continuous-deployment-with-jenkins--cms-21511)
- [backup](https://www.fourkitchens.com/blog/article/trigger-jenkins-builds-pushing-github)
- [another backup](https://learning-continuous-deployment.github.io/jenkins/github/2015/04/17/github-jenkins/)

Inside the Jenkins browser thing:
1. Click the New Item button. Select Build a free-style software project, and click the button labeled OK.
2. Add our project's GitHub URL to the GitHub project box
3. Select the Git option under Source Code Management.
4. Add the URL to our GitHub project repo to the Repository URL
5. Enable Build when a change is pushed to GitHub
6. Click the Add build step drop-down, and select Execute shell
7. I added the build step: `nosetests dg1d`
8. Click Save

Next go to the Github repo:
1. Setting > Click the Services tab, and then the Add service drop-down. Select the Jenkins (GitHub plugin) service.
2. Add the following as the Jenkins hook url: http://JENKINS.SERVER.IP.ADDRESS:8080/github-webhook/
3. Add Service

Test the setup:
1. make a small commit (maybe something that breaks the tests)
2. push 
3. look at the jenkins page (you should see a failed build)
4. revert your commit and push again
5. you should see a successful build

### Configuring for python nosetest
Reference: 
- [try this](http://www.alexconrad.org/2011/10/jenkins-and-python.html)
- [or this](http://nose.readthedocs.io/en/latest/plugins/xunit.html)

Follow these steps:
1. Navigate to the project > Configure > Build 
2. Modify the nosetest build command to contain `--with-xunit`
3. In post-build actions, add the "Publish JUnit test result report" and add the `nosetests.xml` file to the line


### Merge-test-commit cycle

In this view, nobody commits to master. Only jenkins does that. We
work on a development branch and push that to Github. Jenkins then
tests the branch and merges it with master IF it passes the
tests. References:
- [this is good](http://andrewtarry.com/jenkins_git_merges/)
- [and this is more like a man page](https://wiki.jenkins-ci.org/display/JENKINS/Git+Plugin#GitPlugin-AdvancedFeatures)

Follow these steps:
1. Navigate to the project > Configure > Source Code Management
2. Branches to build should be `develop`
3. Add an "Additional Behavior" called "Merge before build" and fill out 
   - name of repository: `origin`
   - branch to merge to: `master`
4. Now create a new post-build action to push the merge back to Github:
   - Git Publisher
   - check "Push Only If Build Succeeds" and "Merge Results"
	
### Add an email notification
1. Navigate to the project > Configure > Post-build action
2. Add the email notification action

