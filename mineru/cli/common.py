# Copyright (c) Opendatalab. All rights reserved.
import io
import json
import os
import copy
import gc
from pathlib import Path
from contextlib import contextmanager

import pypdfium2 as pdfium
from loguru import logger

from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox
from mineru.utils.enum_class import MakeMode
from mineru.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from mineru.backend.vlm.vlm_middle_json_mkcontent import union_make as vlm_union_make
from mineru.backend.vlm.vlm_analyze import doc_analyze as vlm_doc_analyze
from mineru.backend.vlm.vlm_analyze import aio_doc_analyze as aio_vlm_doc_analyze

# Use simple suffix list (compatible, doesn't require magika)
pdf_suffixes = [".pdf"]
image_suffixes = [".png", ".jpeg", ".jpg", ".webp", ".gif", ".bmp", ".jp2"]


@contextmanager
def safe_pdf_document(pdf_bytes):
    """
    Context manager for safe pypdfium2 usage.
    Ensures proper cleanup even on exceptions to prevent C++ memory corruption.
    """
    pdf = None
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        yield pdf
    except Exception as e:
        logger.error(f"Error in pypdfium2 PDF document: {e}")
        raise
    finally:
        if pdf is not None:
            try:
                pdf.close()
            except Exception as e:
                logger.warning(f"Failed to close PDF document: {e}")
        # Force garbage collection to ensure C++ objects are destroyed
        gc.collect()


def read_fn(path):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        # Use simple suffix detection (compatible with current setup)
        if path.suffix.lower() in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif path.suffix.lower() in pdf_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {path.suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method, create_images_dir=True):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    if create_images_dir:
        os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir


def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    """
    Convert PDF bytes to bytes using pypdfium2 with safe resource management.

    IMPORTANT: This function now uses a context manager to prevent C++ memory corruption
    that was causing crashes every 5-6 files ("Pure virtual function called!" error).
    """
    output_pdf = None
    output_buffer = None

    try:
        # Use safe context manager for PDF document
        with safe_pdf_document(pdf_bytes) as pdf:
            # Validate and determine end page
            max_pages = len(pdf)

            # Handle end_page_id
            if end_page_id is None or end_page_id < 0:
                end_page_id = max_pages - 1
            elif end_page_id > max_pages - 1:
                logger.warning(f"end_page_id ({end_page_id}) is out of range (max: {max_pages-1}), using max pages")
                end_page_id = max_pages - 1

            # Validate start_page_id
            if start_page_id < 0:
                logger.warning(f"start_page_id ({start_page_id}) is negative, using 0")
                start_page_id = 0
            elif start_page_id > end_page_id:
                logger.error(f"Invalid page range: start={start_page_id}, end={end_page_id}, returning original")
                return pdf_bytes

            # Create output PDF and import pages
            try:
                output_pdf = pdfium.PdfDocument.new()
                page_indices = list(range(start_page_id, end_page_id + 1))

                # Import pages
                output_pdf.import_pages(pdf, page_indices)

                # Save to buffer
                output_buffer = io.BytesIO()
                output_pdf.save(output_buffer)
                output_bytes = output_buffer.getvalue()

                return output_bytes

            finally:
                # Ensure output_pdf is closed
                if output_pdf is not None:
                    try:
                        output_pdf.close()
                    except Exception as e:
                        logger.warning(f"Failed to close output PDF: {e}")

                # Clear buffer
                if output_buffer is not None:
                    try:
                        output_buffer.close()
                    except:
                        pass

    except Exception as e:
        logger.error(f"Failed to convert PDF bytes: {e}")
        raise
    finally:
        # Aggressive cleanup to prevent memory leaks
        gc.collect()


def _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    """准备处理PDF字节数据"""
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        is_pipeline=True,
        images_enable=False,
):
    f_draw_line_sort_bbox = False
    from mineru.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        md_writer.write(
            f"{pdf_file_name}_origin.pdf",
            pdf_bytes,
        )

    if f_draw_line_sort_bbox:
        draw_line_sort_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_line_sort.pdf")

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir, images_enable)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        make_func = pipeline_union_make if is_pipeline else vlm_union_make
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir, images_enable)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        if is_pipeline:
            md_writer.write_string(
                f"{pdf_file_name}_model.json",
                json.dumps(model_output, ensure_ascii=False, indent=4),
            )
        else:
            output_text = ("\n" + "-" * 50 + "\n").join(model_output)
            md_writer.write_string(
                f"{pdf_file_name}_model_output.txt",
                output_text,
            )

    logger.info(f"local output dir is {local_md_dir}")


def _process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        images_enable=False,
):
    """处理pipeline后端逻辑"""
    from mineru.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from mineru.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze

    infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list = (
        pipeline_doc_analyze(
            pdf_bytes_list, p_lang_list, parse_method=parse_method,
            formula_enable=p_formula_enable, table_enable=p_table_enable
        )
    )

    for idx, model_list in enumerate(infer_results):
        model_json = copy.deepcopy(model_list)
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method, create_images_dir=images_enable)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        images_list = all_image_lists[idx]
        pdf_doc = all_pdf_docs[idx]
        _lang = lang_list[idx]
        _ocr_enable = ocr_enabled_list[idx]

        middle_json = pipeline_result_to_middle_json(
            model_list, images_list, pdf_doc, image_writer,
            _lang, _ocr_enable, p_formula_enable
        )

        pdf_info = middle_json["pdf_info"]
        pdf_bytes = pdf_bytes_list[idx]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, model_json, is_pipeline=True, images_enable=images_enable
        )


async def _async_process_vlm(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        server_url=None,
        images_enable=False,
        **kwargs,
):
    """异步处理VLM后端逻辑"""
    parse_method = "vlm"
    f_draw_span_bbox = False
    if not backend.endswith("client"):
        server_url = None

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method, create_images_dir=images_enable)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json, infer_result = await aio_vlm_doc_analyze(
            pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url, **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, is_pipeline=False, images_enable=images_enable
        )


def _process_vlm(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        backend,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
        server_url=None,
        images_enable=False,
        **kwargs,
):
    """同步处理VLM后端逻辑"""
    parse_method = "vlm"
    f_draw_span_bbox = False
    if not backend.endswith("client"):
        server_url = None

    for idx, pdf_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[idx]
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method, create_images_dir=images_enable)
        image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)

        middle_json, infer_result = vlm_doc_analyze(
            pdf_bytes, image_writer=image_writer, backend=backend, server_url=server_url, **kwargs,
        )

        pdf_info = middle_json["pdf_info"]

        _process_output(
            pdf_info, pdf_bytes, pdf_file_name, local_md_dir, local_image_dir,
            md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf,
            f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
            f_make_md_mode, middle_json, infer_result, is_pipeline=False, images_enable=images_enable
        )


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        images_enable=False,
        **kwargs,
):
    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode, images_enable
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
        os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

        _process_vlm(
            output_dir, pdf_file_names, pdf_bytes_list, backend,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            server_url, images_enable, **kwargs,
        )


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        images_enable=False,
        **kwargs,
):
    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        # pipeline模式暂不支持异步，使用同步处理方式
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode, images_enable
        )
    else:
        if backend.startswith("vlm-"):
            backend = backend[4:]

        os.environ['MINERU_VLM_FORMULA_ENABLE'] = str(formula_enable)
        os.environ['MINERU_VLM_TABLE_ENABLE'] = str(table_enable)

        await _async_process_vlm(
            output_dir, pdf_file_names, pdf_bytes_list, backend,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
            server_url, images_enable, **kwargs,
        )



if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
       do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))],["ch"],
                end_page_id=10,
                backend='vlm-huggingface'
                # backend = 'pipeline'
                )
    except Exception as e:
        logger.exception(e)
