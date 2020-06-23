# Azure Function app in Docker form

* Initiated the functions folder with ``func init``
* Run locally with ``func run``
* Publish (after pushing to docker) with ``func publish``
* Set up, either in the Portal or CLI an Azure Functions app using Python
* Specify, in the CLI or Portal the "custom image", url, pw etc. for your image
* Run ``docker build`` from the root to allow copying of the ``flass`` files to be built.
* With a built and tagged image, run ``docker push``


