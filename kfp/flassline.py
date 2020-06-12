import kfp.compiler
import kfp.dsl
import kfp


@kfp.dsl.pipeline(name="Flass onestep", description="Run the flass in one go")
def flass_onestep(dataset, subset, model_type, epochs, batch_size):
    return kfp.dsl.ContainerOp(
        name="Flass - all in one",
        image="kminaister/flass:1.7.0",
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
        file_outputs={"mlflow_output": "/mlflow/"},
        output_artifact_paths={"mlflow_artifacts_path": "/mlflow/"},
    )


if __name__ == "__main__":
    # Compiling the pipeline
    kfp.compiler.Compiler().compile(flass_onestep, __file__ + ".yaml")
