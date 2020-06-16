import kfp.compiler
import kfp.dsl
import kfp


@kfp.dsl.pipeline(name="Flass onestep", description="Run the flass in one go")
def flass_onestep(
    dataset="mnist", subset=-1, model_type="kerasconv", epochs=3, batch_size=100
):
    return kfp.dsl.ContainerOp(
        name="Flass - all in one",
        image="docker.io/kminaister/flass:0.1.9",
        arguments=[
            "flass",
            "--dataset",
            dataset,
            "--subset",
            subset,
            "--model-type",
            model_type,
            "--epochs",
            epochs,
            "--batch-size",
            batch_size,
        ],
        output_artifact_paths={"classification-report-json": "/home/appuser/classification_report.json"},
    )


if __name__ == "__main__":
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(flass_onestep, __file__ + ".yaml")
