# Azure Function app in Docker form

* Initiated the functions folder with ``func init``
* Build and tag docker image (in this folder)
* Run locally with ``func run``
* Publish (after pushing to docker) with ``func publish``
* Set up, either in the Portal or CLI an Azure Functions app using Python
  
* Specify, in the CLI or Portal the "custom image", url, pw etc. for your image
  * Once created, see the function apps docker setting with: 
  ``az functionapp config container show --name name-of-function-app --resource-group name-of-resource-group``
  * Modify with commands like ``az functionapp config container set --docker-custom-image-name kminaister/flass-function-server:1.0.0 --name name-of-function-app --resource-group name-of-resource-group``  
* Run ``docker build`` from the root to allow copying of the ``flass`` files to be built.
* With a built and tagged image, run ``docker push``


