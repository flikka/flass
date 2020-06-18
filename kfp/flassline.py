import kfp.compiler
import kfp.dsl
import kfp


@kfp.dsl.pipeline(name="Flass onestep", description="Run the flass in one go")
def flass_onestep(
    dataset="mnist",
    subset: int = -1,
    model_type="kerasconv",
    epochs: int = 3,
    batch_size: int = 100,
):
    return kfp.dsl.ContainerOp(
        name="Flass - all in one",
        image="docker.io/kminaister/flass:0.1.9",
        command=["flass"],
        arguments=[
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
        output_artifact_paths={
            "classification-report-json": "/tmp/classification_report.json"
        },
    )


if __name__ == "__main__":
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(flass_onestep, __file__ + ".yaml")
