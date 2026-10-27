import { getOrganizationReportEvidencePackagePdf } from "../../../../../lib/api";
import { readAppSession } from "../../../../../lib/session";

type ReportEvidencePackagePdfRouteContext = {
  params: Promise<{ reportId: string }>;
};

export async function GET(
  _request: Request,
  { params }: ReportEvidencePackagePdfRouteContext
): Promise<Response> {
  const { reportId } = await params;
  const session = await readAppSession();

  if (!session.userId || !session.organizationId) {
    return new Response("Sign in to download this evidence package.", { status: 401 });
  }

  try {
    const exportFile = await getOrganizationReportEvidencePackagePdf(
      session.userId,
      session.organizationId,
      reportId,
      session.authToken
    );
    return new Response(exportFile.content, {
      headers: {
        "Cache-Control": "no-store",
        "Content-Disposition": `attachment; filename="${exportFile.filename}"`,
        "Content-Type": exportFile.content_type
      }
    });
  } catch {
    return new Response("Report evidence package PDF unavailable.", { status: 404 });
  }
}
